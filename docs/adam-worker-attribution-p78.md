# P78: Adam contribution and worker event attribution

## Changed questions and frozen boundaries

[P77](phase-step-p77.md) found no joint spectral gain from a larger phase-readout
step. Its native changes and aggregate worker counters do not identify which
optimizer source or scheduling stage caused the observed behavior. P78 adds
attribution, not a longer unchanged training run or a retry of failed timings.

The optimizer audit replays all sixteen P77 two-update trajectories (32 previously
studied updates). Capturing derivatives must leave every final coordinate model
and persistent Adam state bitwise identical to its frozen P77 endpoint. Only the
first update, which shares the initial state between arms, receives causal
numerator/denominator attribution. Keep all four origins and both physical gains.
Use only the same eight fit draws; no transfer rescoring or new candidate export.

In canonical parameter units, split Adam's first moment into inherited momentum,
the current old-objective gradient and the effective routed-minus-old gradient.
Use the same routed denominator for this split. Separately decompose the routed
minus old step into its extra numerator and changed second-moment denominator.
The effective extra includes FP32 accumulation differences; it is not an exact
symbolic isolated auxiliary derivative. Report base, hidden phase layer and phase
readout independently, with gradient norms and directional derivatives. No moment
reset or altered optimizer is used as a treatment.

FP64 algebra has an independent actual-Adam unit control. Reconstruction of the
real FP32 parameter write must fit two local FP32 ULP plus
`16 * float32_epsilon * abs(analytic_step)`. This is a numerical rounding check,
not an audio threshold. Active global clipping stops this attribution rather
than silently ignoring its coupling. All endpoint states must remain exact.
Evidence: `data/runs/adam-contributions-p78-20261116/`.

Runtime work adds an explicitly enabled bounded event recorder around the shared
callback/worker. It must preserve ordinary dispatch, fixed delay, fallback,
history and queue capacity. Allocation happens before playback and serialization
after worker join. Instrumentation changes the measured binary and may perturb
timing; no new trace may erase P77's fourteen misses, processing overrun or final
harness failure. Functional fault controls precede a separately fixed trace budget.

## Registered runtime trace

`training/scripts/trace_context_worker.py` freezes sixteen once-only cases:
all four P74 parent graphs, the existing P77 synthetic 44.1 kHz stereo input and
stored 48 kHz mono mid music view, and alternating QoS/RT request order. Each
case has 1,100 callbacks; retain the existing whole-normal and post-100-hop rules.
This narrows coverage deliberately for event attribution and cannot certify
all rate/channel/music combinations or compare scheduling-policy superiority.
Evidence: `data/runs/context-events-p78-20261116/`.

The trace is opt-in, fixed-capacity and allocated before playback. Callback and
worker write bounded atomic slots; JSON is created after join. It records input
submission/rejection, worker take/start/finish/terminal action, reset/warmup,
callback decision and packet consumption. Cross-check every count and processing
maximum against final counters, and keep nominal host arrivals distinct from
actual callback decisions. Terminal timestamps follow enqueue/expiry decisions;
publication versus consumer races can remain ambiguous. Wall events do not
identify thread CPU time or an OS scheduling root cause. All sixteen cases must
execute before scoring; no retries or threshold changes.

Functional controls cover unchanged PCM, per-hop clock ordering, dry queue-full
fallback, an actual worker blocked past presentation, gap resets, fixed-capacity
overflow, callback allocation counts, default and no-default builds. Neither
these controls nor a favorable short trace certify physical-device continuity.

## Completed optimizer attribution

All sixteen final coordinate models, canonical models and Adam states match
P77 bitwise. No clipping is active. The analytic first-step decomposition fits
the registered FP32 rounding bound; maximum parameter-write difference is
5.959e-8. Original target/reference, data exposure and failure decisions are
unchanged. Full tensors and all groups are retained in `first-step.pth` and
`assessment.json`; no changed candidate is exported or scored.

For the readout, inherited-moment step norms are 2.89–3.32 times the current old
gradient component. Yet the inherited component points **down** the auxiliary
in all four origins, as does the complete readout update. Its size is not proof
that momentum blocks learning. In the hidden phase layer the inherited component
opposes the auxiliary in two origins; one origin's net hidden-layer update is
locally adverse. This is a layer-specific local conflict, not a reason to reset
the entire optimizer.

The effective auxiliary gradient is zero in the backbone, confirming routing.
This is not zero sensitivity of the auxiliary to the backbone: the old objective
still changes base parameters, which feed both magnitude and phase predictions.
In the phase readout it is about 11.1–17.0% of the old gradient norm. Its incremental
Adam step is not cancelled by the changed second-moment denominator: the retained
increment-to-sum-of-components norm ratio is 0.977–0.996 across head layers.
Therefore neither a swallowed auxiliary nor blanket momentum obstruction explains
P77. The current component is small relative to accumulated motion, but its
direction reaches the target. P77's remaining spectral tradeoff must be assessed
at actual output and at the relevant layers; no longer run or global moment reset
follows from this diagnostic alone.

## Completed worker attribution

All sixteen once-only cases complete successfully, with 17,600 input hops and
16,000 post-warmup callbacks. Every event/counter cross-check passes, including
the final post-join processing and queue-wait maxima. Fifteen software screens
pass and one fails. There are three presentation misses in two cases; one case
has a single allowed miss and therefore still passes the existing availability
ratio. No processing call exceeds its hop period or 5 ms in this trace grid.
This narrower, instrumented run does not replace any P77 failure.

| Observed quantity | Result |
| --- | --- |
| Worker processing maximum | 2.459625 ms |
| Callback p99 / maximum, worst case | 0.039005 / 0.089667 ms |
| Queue-wait / producer-lateness maximum | 1.113625 / 2.961837 ms |
| Longest presentation-miss run | Two 44.1 kHz hops, 5.805 ms |
| Peak process RSS | 42.297–44.625 MiB |
| Process one-core-equivalent CPU | 13.58–39.88%, including load, trace serialization and IO |
| Output | All finite/bounded, 140,800 frames per case, fixed 256-frame delay |

The exact counterexamples, under `cases/07/` and `cases/12/`, are:

- Case 07, seed 20261013 phase, synthetic 44.1 kHz stereo, QoS: callback 133
  begins 2.961837 ms late. The fixed-clock replay then catches up with callback
  134 only 0.014041 ms later. Input 133 has only 0.009666 ms from submission to
  that callback's beginning and arrives after its nominal presentation time. The
  worker supersedes that obsolete input. Callback 134 misses; the next accepted
  input resets history, so callback 135 rejects its continuity-invalid result.
  That second result's processing takes 2.252250 ms within an available 3.557666 ms
  window: this miss is a required reset response, not slow inference. Total
  dry-history warmup rises from 80 to 160 hops. The two misses fail the unchanged
  per-case ratio; the extra history recovery remains separate from the burst.
- Case 12, seed 20261013 magnitude, stored 48 kHz mono mid music, QoS: callback
  544 begins 2.369458 ms late. Input 544 waits 0.039208 ms and takes 2.459625 ms
  to process, but callback 545 starts only 1.892333 ms after its submission.
  Processing is still in progress when the callback decides, and the worker
  later discards the expired result. The isolated miss fits the existing ratio
  allowance, but it still remains in the trace. Being faster than 5 ms and even
  faster than a nominal hop cannot recover time already lost to late submission.

The first case establishes a catch-up burst in this **host simulation**, not a
measured physical-device xrun. The second proves an insufficient actual processing
window; wall timestamps cannot attribute its compute time to CPU work versus
preemption. All eight requested RT policies are accepted at startup, which does
not establish their later state or superiority. The protocol has no simultaneous
untraced control, physical audio callback, device workgroup or listening evidence.

## Decision and validation

Do not reset all Adam moments or increase the global learning rate on this evidence.
The next training-side question is how the finite hidden/readout/base changes
combine at the output: local gradient improvement does not prove improved spectral
shape after reconstruction. A changed experiment must hold the same original
targets and matched endpoints, isolate these contributions, and keep P77's failed
joint spectral gates intact. No full training or weight promotion follows here.

Runtime work should distinguish actual host presentation windows from nominal
fixed-clock replay, with explicit late-producer fault controls or a real host
integration. Do not remove gap resets, shorten history or add buffering to make
these cases pass. Retain both nominal and actual timing; a slower artificial
producer that hides the short windows is not a deadline fix.

The core library/integration suite passes with tracing, including callback
allocation and controlled delayed-worker tests; no-default-feature checks pass.
Eight focused Python algebra/counter/ambiguity checks pass, along with Clippy,
Ruff and documentation checks. The accelerated 30-minute audio test remains
ignored; no physical-device long run is claimed. Both complete receipts bind
their frozen sources and raw evidence. Defaults remain unchanged.
