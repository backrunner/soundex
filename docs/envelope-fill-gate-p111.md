# P111 soft fill gate for clean-input preservation

[P110](envelope-native-p110.md) found that native Envelope mode beats the native
P74 parent on missing-band restoration but raises clean-input preserved LSD by
+0.90/+0.38 dB. The ported detector rarely bypasses real music, so the fill runs
on lossless input. P111 adds a DSP-side gate with no retraining and re-tests
natively. No default, release or listening decision follows.

## Absolute scale of the P110 failure

`AGENTS.md` requires calibrated absolute errors beside relative scores for weak
bands, and P110 did not report them. On the 16 clean works, the >4 kHz part of
output minus input is -88 dBFS for P74 and -80/-83 dBFS for ungated P110
(recording means of dB). The clean high band itself is -43 dBFS median. The
P110 regression is therefore mostly relative error in near-silent high bins.

## Gate (commit `f18bd82`)

Each band's fill is scaled linearly in dB of 10 log10(fill / present band
energy), from 0 at 0 dB to 1 at 6 dB. The bounds come from the target
definition and were fixed before scoring:
- **0 dB:** a fill there would only double the band's existing energy.
- **6 dB:** present content is at most 20% of the target.
- **Separation:** the deficit target floors identity bands at -30 dB, so clean
  bands sit far below the gate and fully missing bands far above it.

The gate travels in the `.sxenv` header as `fill_gate_db`. Artifacts without
the key behave as before. The Python proxy and the Rust fill implement the same
gate, with Rust tests for parsing, validation and gated/missing-band behavior.

## Registered rules

`data/runs/envelope-gate-p111-20261124/registration.json`; both P109 seeds,
SmoothKnee limiter, all 64 P88 cases:
1. **Correspondence:** native output within 0.1 dB of the gated, limited proxy.
2. **Primary:** in both seeds, clean preserved LSD versus the native P74
   parent is not an increase, and missing-region ERB RMS versus P74 has an
   interval below zero.
3. **Reported:** gated versus ungated, absolute clean high-band error, codecs.

## Results

**Rule 1: met.** Recording means agree to 0.000 dB; waveform error-to-signal is
-136 dB median and -87 dB worst.

| Path | Missing LSD | ERB RMS | ERB bias | Temporal corr | Paired preserved LSD | Clean preserved LSD | Clean HF error dBFS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P74 parent | 14.57 | 14.44 | -9.60 | 0.569 | 7.15 | 0.70 | -87.8 |
| Ungated, seed 1 / 2 | 13.11 / 13.05 | 12.57 / 12.53 | -7.73 / -8.08 | 0.578 / 0.577 | 7.41 / 7.35 | 1.61 / 1.08 | -80.3 / -82.7 |
| **Gated, seed 1 / 2** | **13.27 / 13.20** | **12.74 / 12.67** | -7.89 / -8.21 | 0.569 / 0.570 | 7.45 / 7.39 | **1.38 / 0.82** | **-92.4 / -97.9** |

| Contrast (seed 1 / seed 2) | ERB RMS dB | Clean preserved LSD dB | Clean HF error dB |
| --- | --- | --- | --- |
| Gated - P74 | -1.70 [-2.88, -0.63] / -1.76 [-2.90, -0.69] | **+0.68 [+0.12, +1.34]** / +0.12 [-0.10, +0.43] | -4.6 [-13.5, +4.4] / -10.1 [-17.3, -3.6] |
| Gated - ungated | +0.17 [+0.10, +0.24] / +0.15 [+0.09, +0.21] | -0.23 [-0.30, -0.16] / -0.26 [-0.34, -0.18] | -12.1 [-21.4, -3.7] / -15.2 [-24.3, -6.2] |

The restoration gain over P74 holds in every codec. ERB RMS against P74:
AAC 12.76/12.20 vs 14.72 dB, MP3 13.39/13.24 vs 14.73 dB, Vorbis 12.07/12.58 vs
13.85 dB.

**Rule 2: not met.** Seed 2 passes both parts. Seed 1 still raises clean
preserved LSD against P74; its interval lies above zero, with 6 of 16 works
adverse. Its envelope part passes.

The gate cut absolute clean high-band error by 12-15 dB, to below the P74
parent's level: below zero in seed 2, unresolved in seed 1. No clean work got
worse against the ungated run. The cost is 0.15-0.17 dB of the envelope-error
gain and 0.04 dB of paired preserved LSD. Seed 1's remaining excess is relative
error in near-silent bins, at an absolute level no higher than P74's. Whether
it is audible is a listening question; this report does not answer it.

## Listening pack (ratings outstanding)

`data/runs/envelope-listening-p111-20261124-v2/listening/index.html` holds 20
trials:
- **16 compressed:** one per work, codecs rotated by seed 111. Conditions:
  untreated, P74, ungated and gated envelope.
- **4 clean-input:** seeded works. Conditions: P74, ungated and gated on
  lossless input.
- **Every trial:** hidden reference, 3.5 kHz anchor, and one common gain.

Seed 1 was used by the fixed lowest-registered-seed rule. It is also the seed
that failed rule 2. The plan was frozen in `frozen-plan.json` before audio was
written, and the key sits outside the listener folder. A first build embedded
a protocol id naming the study topic; it was superseded before any rating
(`...-20261124/superseded.json`).

## Interpretation

The gated envelope mode keeps a 1.7 dB envelope-error and 1.3 dB LSD advantage
over the preserved P74 parent in every codec. It costs well under 1 ms per hop
(P110), and its absolute clean high-band error is at or below P74's. It fails
its registered relative clean criterion in one of two seeds. It stays
experimental, and P74 stays the preserved native reference until listener
ratings or a registered follow-up resolve clean preservation.

## Evidence

`data/runs/envelope-gate-p111-20261124/` holds the registration, gated artifacts,
renders, records, summary, contrasts and correspondence; the binary is in
`data/runs/envelope-gate-p111-assets-20261124/`. Code:
`training/scripts/run_envelope_gate_p111.py` and
`training/scripts/build_p111_listening.py`.
