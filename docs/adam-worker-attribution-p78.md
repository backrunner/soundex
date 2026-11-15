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

## Status

The algebra controls pass. Exact endpoint replay and bounded runtime event
instrumentation are in progress. No changed objective, full training, weight
promotion, new quality claim or device qualification is authorized by this audit.
