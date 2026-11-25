"""Conservative rights screen for training/evaluation candidates (P115 policy).

Accept only CC0 1.0 or CC BY (2.0-4.0, including ported versions) item licences
that match the catalog grant, and exclude any item whose publisher metadata or
file entry shows: another licence restriction (NC/ND/SA, all rights reserved),
any mention of AI/ML/data-mining terms, or indicators of third-party material
(remix, sample, cover, mash-up, bootleg, edit, medley, tribute, parody, a
cappella, interpolation, Freesound, featured artists). Creators with a recorded
AI-training prohibition are excluded. False positives are accepted by design.
Passing this screen reduces, but cannot eliminate, rights risk: a CC grant covers
only rights the licensor holds. Maintainer review still governs releases.
SPDX-License-Identifier: Apache-2.0
"""

import json
import re
from pathlib import Path

POLICY_VERSION = "soundex-rights-screen-v3"  # v2: cover art; v3: creator credits and attribution.
ALLOWED_LICENSE = re.compile(
    r"^https?://creativecommons\.org/(licenses/by/(2\.0|2\.5|3\.0|4\.0)(/[a-z]{2})?/?"
    r"|publicdomain/zero/1\.0/?)$"
)
RESTRICTED = re.compile(
    r"non-?commercial|\bby-nc|\bnc\b|no ?derivativ|\bby-nd|\bnd\b|share-?alike|\bby-sa"
    r"|all rights reserved",
    re.I,
)
AI_TERMS = re.compile(
    r"\bai\b|\ba\.i\.|artificial intelligence|machine learning|\bneural\b|\bdataset"
    r"|data ?mining|text and data|\btdm\b|\bml\b|\btrain(ing|ed)? (a |an )?(model|ai)",
    re.I,
)
THIRD_PARTY = re.compile(
    r"remix|\brmx\b|mash-?up|bootleg|\bcovers?\b(?!\s*(art|artwork|photo|photograph|design|image"
    r"|illustration|painting|picture|by\b))|cover version|\bsampl(e|es|ed|ing)\b|freesound"
    r"|\bfeat\b|\bfeat\.|featuring|\bft\.|\bre-?edit|\bedit\b|\bmedley\b|\btribute\b"
    r"|\bparody\b|a ?cap+ella|interpolat|\bflip(ped)?\b",
    re.I,
)
BLOCKED_CREATORS = re.compile(r"loyalty freak|monplaisir|komiku", re.I)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)


def screen(row, metadata):
    """Return a list of exclusion reasons ([] = passes) for one catalog row."""
    reasons = []
    item = metadata.get("metadata", {})
    license_url = _text(item.get("licenseurl")).strip()
    if not ALLOWED_LICENSE.match(license_url):
        reasons.append(f"item_license_not_cc0_or_cc_by:{license_url or 'none'}")
    catalog = row.get("license_url", "").rstrip("/").replace("https://", "http://")
    if license_url.rstrip("/").replace("https://", "http://") != catalog:
        reasons.append("catalog_license_differs_from_item")
    filename = row["id"].split("/", 1)[1] if "/" in row["id"] else row["id"]
    entry = next((f for f in metadata.get("files", []) if f.get("name") == filename), {})
    fields = " | ".join(
        _text(x)
        for x in (
            item.get("title"),
            item.get("description"),
            item.get("subject"),
            item.get("notes"),
            item.get("rights"),
            filename,
            entry.get("title"),
            entry.get("comment"),
            entry.get("album"),
            entry.get("creator"),
            row.get("recording_title"),
            row.get("album"),
        )
    )
    creators = " | ".join(
        _text(x) for x in (item.get("creator"), row.get("author"), entry.get("creator"))
    )
    if RESTRICTED.search(fields):
        reasons.append("restriction_text:" + RESTRICTED.search(fields).group(0).lower())
    if AI_TERMS.search(fields):
        reasons.append("ai_or_data_terms:" + AI_TERMS.search(fields).group(0).lower())
    if THIRD_PARTY.search(fields):
        reasons.append("third_party_indicator:" + THIRD_PARTY.search(fields).group(0).lower())
    if BLOCKED_CREATORS.search(creators):
        reasons.append("blocked_creator")
    if THIRD_PARTY.search(creators):
        reasons.append("third_party_in_creator:" + THIRD_PARTY.search(creators).group(0).lower())
    named = [c for c in (_text(item.get("creator")), _text(row.get("author"))) if c.strip()]
    if not any(
        c.strip().lower() not in ("none", "unknown", "various", "various artists") for c in named
    ):
        reasons.append("no_attributable_creator")
    if entry and entry.get("source") != "original":
        reasons.append("not_publisher_original_file")
    return reasons


def screen_rows(rows):
    """Screen rows with a local `source_metadata_path`; returns (passed, excluded)."""
    passed, excluded = [], []
    for row in rows:
        path = row.get("source_metadata_path")
        if not path or not Path(path).is_file():
            excluded.append({"id": row["id"], "reasons": ["missing_publisher_metadata"]})
            continue
        reasons = screen(row, json.loads(Path(path).read_text()))
        if reasons:
            excluded.append({"id": row["id"], "author": row.get("author"), "reasons": reasons})
        else:
            passed.append(row)
    return passed, excluded
