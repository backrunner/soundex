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
In the phase readout it is about 11.1–17.0% of the old gradient norm. Its incremental
Adam step is not cancelled by the changed second-moment denominator: the retained
increment-to-sum-of-components norm ratio is 0.977–0.996 across head layers.
Therefore neither a swallowed auxiliary nor blanket momentum obstruction explains
P77. The current component is small relative to accumulated motion, but its
direction reaches the target. P77's remaining spectral tradeoff must be assessed
at actual output and at the relevant layers; no longer run or global moment reset
follows from this diagnostic alone.

The registered runtime trace is pending. Defaults remain unchanged; there is no
new training, restoration-quality or physical-device qualification claim.
