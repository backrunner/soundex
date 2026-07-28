# Mix-only download & disk plan (SoundEx)

Goal: put **full-band mixture audio only** on disk. Never keep stems / RAW / MIDI
for training.

Official archives often ship multitracks. Prefer **extract-filter** or
**download-then-prune** so peak disk use is temporary.

## Recommended disk layout

| Mount | Suggested size | Contents |
|-------|----------------|----------|
| `/data` (or workspace data disk) | **≥ 200 GB free** (comfortable **250–300 GB**) | raw mixes + `processed/` + headroom |
| System / CUDA | separate | PyTorch, caches |

If you only train **MUSDB** first: **≥ 40 GB** is enough.  
Full **MUSDB + Slakh + MedleyDB** mix-only + processed: plan **~200 GB**.

---

## Size budget (approximate)

### A. Raw mixes only (after filtering)

| Dataset | Full multitrack download (typical) | **Mix-only kept** | Notes |
|---------|------------------------------------|-------------------|--------|
| MUSDB18-HQ | ~**23 GB** zip/tree | ~**4–6 GB** | Keep `mixture.wav` only |
| Slakh2100 FLAC | ~**100–110 GB** (stems+mix) | ~**15–30 GB** | Keep `mix.flac` only; stems are most of the size |
| MedleyDB V1+V2 | often **tens of GB** with STEMS+RAW | ~**6–12 GB** | Keep `*_MIX.wav` (+ optional metadata yaml) |
| **Subtotal raw mixes** | | **~25–50 GB** | |

### B. SoundEx `processed/` pairs

Preprocessors preserve stereo through the codec round-trip, correct codec delay, then derive
configured mono/left/right/mid/side examples. They publish a checksummed MP3 CBR/VBR, AAC-LC,
and Vorbis matrix at 44.1/48 kHz into an immutable directory:

```text
processed/<corpus>-<recipe-hash>/
  recipe.json
  manifest.jsonl
  summary.json
  audio/{train,validation,test}/*_{clean,degraded}.wav
```

The exact size is recipe-dependent and larger than the former MP3-only estimate. Run a small
recipe on representative tracks, inspect `summary.json`, and extrapolate before full generation:

| Dataset | Order of magnitude (default segments) |
|---------|----------------------------------------|
| MUSDB | Depends on codec/rate/channel-role strata |
| Slakh | Largest; bound with `data.recipe.segment.count_per_track` |
| MedleyDB | Depends on available stereo mixes |
| **Subtotal processed** | Measure from the resolved recipe before a full run |

### C. Training runtime

| Item | Size |
|------|------|
| Checkpoints every 10 epochs | ~**0.5–2 GB** (keep last few) |
| TensorBoard logs | small |
| ffmpeg /tmp during preprocess | spikes **+5–20 GB** if many parallel jobs |

### Peak vs steady state

```
Peak (unpack full Slakh then prune):   often 120–180 GB transient
Steady (mix-only raw + processed):     ~60–120 GB typical
Recommended free space before start:   250 GB+
```

---

## Titan Xp 12 GB — training feasibility

**Yes, sufficient** for the current SoundEx model (~1.3M generator + small MSD).

| Setting | Recommendation on Titan Xp 12 GB |
|---------|-----------------------------------|
| `batch_size` | **16** default is fine; if OOM use **8** |
| AMP / fp16 | optional; not required at this size |
| `context_frames` | 8 (default) — tiny activations |
| CUDA wheels | Target **torch 2.12.1** on **cu126** (safer for Pascal) or **cu130** (CUDA 13.0 family, modern GPUs). See `environment.md`. |
| Driver | Must satisfy the chosen PyTorch wheel (cu130 needs a newer driver than cu126) |

Rough VRAM (fp32, BS=16, GAN phase): **~2–6 GB** peak in earlier estimates — well under 12 GB.  
Bottlenecks will be **disk I/O** and **CPU ffmpeg preprocess**, not VRAM.

Throughput: Pascal is slower than 3090; full MUSDB+Slakh+MedleyDB × 200 epochs may take **longer wall-clock** (still on the order of tens of hours to a few days depending on data volume), but **fits in memory**.

---

## Download recipes (mix-focused)

### 1) MUSDB18-HQ

1. Download from Zenodo: https://zenodo.org/records/3338373 (~22.7 GB).  
2. Unpack.  
3. Prune stems immediately:

```bash
python training/scripts/filter_mix_only.py \
  --root /data/musdb18-hq \
  --dataset musdb \
  --delete
```

4. Preprocess:

```bash
python training/data/preprocess_musdb.py \
  --data-root /data/musdb18-hq \
  --output-dir /data/musdb18-hq/processed
```

### 2) Slakh2100 (largest savings if you drop stems)

1. Prefer **Slakh2100-flac-redux** from Zenodo (see http://www.slakh.com/).  
2. If the archive is a single tarball, extract **only mix files** when possible:

```bash
# Example pattern — adjust archive name/paths to the file you downloaded
mkdir -p /data/slakh2100
tar -xf slakh2100_flac_redux.tar.gz -C /data/slakh2100 \
  --wildcards --no-anchored '*/mix.flac' \
  --wildcards --no-anchored '*/metadata.yaml'
```

If selective extract is awkward, unpack then prune:

```bash
python training/scripts/filter_mix_only.py \
  --root /data/slakh2100 \
  --dataset slakh \
  --delete
```

3. Confirm no `stems/` trees remain:

```bash
find /data/slakh2100 -type d -name stems | head
# should print nothing
```

4. Preprocess:

```bash
python training/data/preprocess_slakh.py \
  --data-root /data/slakh2100 \
  --output-dir /data/slakh2100/processed
```

### 3) MedleyDB

1. Obtain V1/V2 audio via official Zenodo request links on https://medleydb.weebly.com/  
2. After unpack:

```bash
python training/scripts/filter_mix_only.py \
  --root /data/medleydb \
  --dataset medleydb \
  --delete
```

3. Preprocess:

```bash
python training/data/preprocess_medleydb.py \
  --data-root /data/medleydb \
  --output-dir /data/medleydb/processed
```

---

## Suggested order on a small disk

1. MUSDB only → train smoke test (Tier C weights if released).  
2. Add Slakh mix-only (biggest quality/quantity jump).  
3. Add MedleyDB mix-only if space remains.  

After each preprocess succeeds, you may delete **raw** mixes and keep only `processed/` to reclaim space (you cannot re-slice without raw). Safer: keep raw mixes, delete only stems forever.

---

## Quick capacity checklist

- [ ] ≥ **200 GB** free before downloading Slakh full archive  
- [ ] Run `filter_mix_only.py --delete` before preprocess  
- [ ] `find … -name stems` empty for Slakh  
- [ ] `find … -iname '*_STEMS'` empty for MedleyDB  
- [ ] `processed/` + raw mixes fit with **≥ 30 GB** spare for checkpoints/tmp  
