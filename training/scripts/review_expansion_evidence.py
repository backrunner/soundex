"""Official-source rights evidence for expansion albums (library review standard).

For each drafted album, find the creator's official Bandcamp catalog (linked from
the publisher item, or the creator name's Bandcamp subdomain), locate the album
whose title matches, and record the licence the creator publishes there, track
overlap, AI/data terms and third-party credit indicators. Every fetched page is
saved with its SHA-256. Outcome per album: verified, contradicted or unverified.
Only "verified" albums may be proposed for acceptance; nothing is accepted here.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import difflib
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.rights_screen import AI_TERMS, THIRD_PARTY  # noqa: E402

REPO = ROOT.parent
UA = "Mozilla/5.0 (compatible; soundex-rights-review/1.0)"
CC = re.compile(r"creativecommons\.org/(licenses/[a-z\-]+/[0-9.]+|publicdomain/[a-z]+/[0-9.]+)")


def norm(text):
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def slug(text):
    return re.sub(r"[^a-z0-9]", "", norm(text))


class Fetcher:
    def __init__(self, evidence):
        self.evidence, self.cache = evidence, {}

    def get(self, url):
        if url in self.cache:
            return self.cache[url]
        time.sleep(1.0)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read()
                final = response.geturl()
        except Exception as error:
            self.cache[url] = (None, None, str(error))
            return self.cache[url]
        digest = hashlib.sha256(body).hexdigest()
        path = self.evidence / f"{digest[:16]}.html"
        path.write_bytes(body)
        record = {"url": url, "final_url": final, "path": str(path), "sha256": digest}
        self.cache[url] = (body.decode("utf-8", "replace"), record, None)
        return self.cache[url]


def bandcamp_roots(description, author, extra=()):
    roots = re.findall(r"https?://([a-z0-9\-]+)\.bandcamp\.com", str(description), re.I)
    roots += [
        r
        for text in extra
        for r in re.findall(r"https?://([a-z0-9\-]+)\.bandcamp\.com", str(text), re.I)
    ]
    guesses = [slug(a) for a in re.split(r"\s*(?:&|,| and | x )\s*", str(author or "")) if slug(a)]
    seen, out = set(), []
    for root in [r.lower() for r in roots] + guesses:
        if root and root not in seen:
            seen.add(root)
            out.append(f"https://{root}.bandcamp.com")
    return out


def catalog_entries(body):
    """(href, displayed title) for every album/track in a Bandcamp /music grid."""
    entries = []
    for href, inner in re.findall(
        r'<a href="(/(?:album|track)/[^"?#]+)"[^>]*>(.*?)</a>', body, re.S
    ):
        title = re.search(r'<p class="title">\s*(.*?)(?:<br|</p>)', inner, re.S)
        text = (
            html.unescape(re.sub(r"<[^>]+>", " ", title.group(1))) if title else href.split("/")[-1]
        )
        entries.append((href, norm(text)))
    for href in set(re.findall(r'href="(/(?:album|track)/[^"?#]+)"', body)):
        entries.append((href, norm(href.split("/")[-1].replace("-", " "))))
    return entries


def album_title(item_title, author):
    title = norm(item_title)
    for part in [norm(author), *map(norm, re.split(r"&|,", str(author or "")))]:
        if part and title.startswith(part + " "):
            title = title[len(part) + 1 :]
    return title.strip()


def review_album(fetch, album, metadata, track_titles, label_texts=()):
    item = metadata.get("metadata", {})
    want = album_title(item.get("title"), album["author"])
    attempts = []
    for root in bandcamp_roots(item.get("description"), album["author"], label_texts):
        body, record, error = fetch.get(root + "/music")
        if body is None:
            attempts.append({"root": root, "error": error})
            continue
        artist_name = (
            norm(re.search(r"<title>(.*?)</title>", body, re.S).group(1))
            if "<title>" in body
            else ""
        )
        best, score = None, 0.0
        for href, title in catalog_entries(body):
            ratio = difflib.SequenceMatcher(None, title, want).ratio() if title and want else 0.0
            if title and want and (title == want or want in title or title in want):
                ratio = max(ratio, 0.95)
            if ratio > score:
                best, score = href, ratio
        if score < 0.85:
            best = None
        if best is None:
            attempts.append(
                {
                    "root": root,
                    "catalog": record,
                    "artist_page_title": artist_name,
                    "match": "album not in official catalog",
                }
            )
            continue
        page, page_record, error = fetch.get(root + best)
        if page is None:
            attempts.append({"root": root, "error": error})
            continue
        licences = sorted(set(CC.findall(page)))
        official_tracks = {
            norm(t) for t in re.findall(r'<span class="track-title">([^<]+)</span>', page)
        }
        overlap = len({norm(t) for t in track_titles} & official_tracks) / max(len(track_titles), 1)
        text = html.unescape(
            re.sub(
                r"<[^>]+>",
                " ",
                " ".join(
                    re.findall(
                        r'class="tralbumData tralbum-(?:about|credits)"[^>]*>(.*?)</div>',
                        page,
                        re.S,
                    )
                ),
            )
        )
        result = {
            "root": root,
            "catalog": record,
            "album_page": page_record,
            "official_licences": licences,
            "all_rights_reserved": "all rights reserved" in page.lower(),
            "track_overlap": round(overlap, 3),
            "official_tracks": len(official_tracks),
            "ai_terms": (AI_TERMS.search(text).group(0) if AI_TERMS.search(text) else None),
            "third_party_credit": (
                THIRD_PARTY.search(text).group(0) if THIRD_PARTY.search(text) else None
            ),
        }
        ok_licence = bool(licences) and all(
            re.fullmatch(r"licenses/by/[0-9.]+|publicdomain/zero/1\.0", x) for x in licences
        )
        if not ok_licence or result["all_rights_reserved"]:
            result["outcome"] = "contradicted"
        elif result["ai_terms"] or result["third_party_credit"] or overlap < 0.5:
            result["outcome"] = "unverified"
        else:
            result["outcome"] = "verified"
        attempts.append(result)
        if result["outcome"] != "unverified" or overlap >= 0.5:
            return result, attempts
    return {"outcome": "unverified", "reason": "no official catalog page matched"}, attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "evidence").mkdir()
    fetch = Fetcher(a.out / "evidence")
    ver = {}
    for p in (
        "data/runs/reserved-panel-p113-20261124/verified-candidates.jsonl",
        "data/runs/reserved-panel-p114-20261124/verified-candidates.jsonl",
    ):
        for line in (REPO / p).read_text().splitlines():
            row = json.loads(line)
            ver[row["audio_sha256"]] = row
    albums = defaultdict(list)
    for line in a.drafts.read_text().splitlines():
        row = json.loads(line)
        albums[row["source_page"]].append(row)
    results = []
    for page, rows in sorted(albums.items()):
        metadata = json.loads(
            Path(ver[rows[0]["audio_sha256"]]["source_metadata_path"]).read_text()
        )
        titles = [
            ver[r["audio_sha256"]].get("recording_title")
            or re.sub(r"\.(flac|wav)$", "", r["id"].split("/")[-1], flags=re.I)
            for r in rows
        ]
        titles = [re.sub(r"^\d+[\s._-]*", "", re.sub(r"^.* - ", "", t)) for t in titles]
        collections = metadata.get("metadata", {}).get("collection") or []
        collections = collections if isinstance(collections, list) else [collections]
        label_texts = []
        for name in collections:
            if name in ("netlabels", "opensource_audio", "audio_music", "podcasts"):
                continue
            body, _record, _error = fetch.get(f"https://archive.org/metadata/{name}")
            if body:
                label_texts.append(json.loads(body).get("metadata", {}).get("description", ""))
        result, attempts = review_album(fetch, rows[0], metadata, titles, label_texts)
        results.append(
            {
                "source_page": page,
                "album": rows[0]["album"],
                "author": rows[0]["author"],
                "works": len(rows),
                "audio_sha256": [r["audio_sha256"] for r in rows],
                "result": result,
                "attempts": attempts,
            }
        )
        print(
            json.dumps({"album": str(rows[0]["album"])[:40], "outcome": result["outcome"]}),
            flush=True,
        )
    (a.out / "album-reviews.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    from collections import Counter

    print(
        json.dumps(
            {
                "albums": dict(Counter(r["result"]["outcome"] for r in results)),
                "works": dict(
                    Counter(r["result"]["outcome"] for r in results for _ in r["audio_sha256"])
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
