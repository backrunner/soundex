# Release-candidate listening protocol

This protocol records subjective evidence; it does not create an automated pass
claim and does not replace the objective release gates.

1. Freeze the ONNX artifact, test manifest hashes, evaluation report, clip list,
   randomization seed, and scoring form before listening begins.
2. Sample clips across every required corpus, codec/quality, sample-rate, and stereo
   role stratum. Use equal-duration excerpts long enough to judge timbre and
   transients, and publish only aggregates permitted by dataset licenses.
3. Present degraded, enhanced, and clean-reference conditions with opaque randomized
   labels. Apply one common level-matching gain per source; never normalize each
   condition independently. Verify the matched loudness tolerance in the record.
4. Use the same playback chain and lossless clip format for every condition. Record
   listener eligibility, headphones/monitors, environment, and excluded trials.
5. Collect separate ratings for high-frequency naturalness, codec artifacts,
   stereo image stability, and overall preference. Include an explicit "no audible
   difference" choice and do not reveal conditions until the dataset is locked.
6. Report anonymized listener/trial counts, per-stratum aggregates and uncertainty,
   exclusions with reasons, randomization method, and any adverse examples. Attach
   the artifact/report/manifest hashes and maintainer sign-off.

A release model card must link the completed record and its anonymized aggregate.
Synthetic fixtures, informal listening notes, or the absence of complaints are not
a completed protocol.
