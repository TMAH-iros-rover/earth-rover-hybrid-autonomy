# best_sam_tp.pt (hf_sam2) vs checkpoint_2.pt (genie_sam_tp) A/B

Single-frame comparison used to validate the new `hf_sam2` backend
(`training/sam_tp_hf_backend.py`) against the existing `genie_sam_tp`
backend on the same real rover front-camera frame.

- `rover_frame.jpg`: extracted with `ffmpeg -frames:v 1` from
  `output_rides_9/ride_27517_20240404140255/recordings/..._uid_s_1000..._20240404140314140.ts`
  (1024x576 RGB, real front-camera footage, not synthetic).
- `hf_sam2_overlay.jpg` / `genie_overlay.jpg`: score heatmap blended over the
  source frame for each backend (`SAM2Transforms`-equivalent preprocessing,
  same bottom-left/bottom-center/bottom-right point prompts).
- `ab_comparison.json`: raw logits stats, sigmoid score stats, and
  `MotionPrimitivePlanner.plan()` output (near-field score/safety, selected
  candidate heading, planner confidence) for both backends against the
  same frame using `configs/default.yaml`'s `planner` section.

## Result summary

Both backends produce a spatially coherent traversability map (ground
region high score, sky/trees/horizon low score) and both leave the
planner in the same qualitative state on this frame: `near_field_safe`,
`trajectory_valid`, selected heading 0 deg, comparable near-field score
(0.93 vs 0.95) and planner confidence (0.92 vs 0.96). Absolute score
values differ between the two checkpoints/backends, which is expected
(different weights, different architecture) and is not itself a failure
per the task's acceptance criteria -- both outputs are valid planner
input.
