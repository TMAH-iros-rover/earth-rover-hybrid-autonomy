# Offline Training Workflow

## Frozen SegFormer v2 and official SAM-TP reproduction

The approved SegFormer-B0 v2 baseline is logically frozen in
`docs/experiments/segformer_b0_v2_frozen_baseline_20260726.md`. Its dataset,
checkpoint, splits, and generated reports remain unchanged outside Git.

The independent official SAM-TP setup, strict config/checkpoint compatibility
gate, single-image benchmark, and deterministic FrodoBots review are documented
in `docs/experiments/sam_tp_reproduction.md`. This is perception-only
reproduction; it does not train SAM-TP or connect it to planning, the SDK, or a
live rover.

## FrodoBots-2K Phase 1 Manifest

Build a bounded three-ride manifest without decoding or extracting video frames:

```bash
.venv/bin/python research/training/build_frodobots_2k_manifest.py \
  --dataset-root ../datasets_2k/output_rides_0 \
  --output-dir ../datasets_2k/manifests/frodobots_2k_phase1/dry_run_3rides \
  --max-rides 3 \
  --control-tolerance-ms 100
```

The command writes `manifest.csv` and `alignment_report.json`. The output must remain outside the immutable `output_rides_0` directory and outside Git. Complete this validation before implementing HLS frame decoding or model training.

On the Dell Ubuntu host, run the complete focused verification with no arguments:

```bash
./research/scripts/verify_phase1_manifest.sh
```

The script uses `$HOME/datasets/output_rides_0`, writes generated files outside Git under `$HOME/datasets/manifests/`, and verifies that the raw dataset metadata is unchanged.

## FrodoBots-2K Phase 2 HLS Verification

On the Dell Ubuntu host, run the lazy HLS loader and visual verification with no arguments:

```bash
./research/scripts/verify_phase2_hls_loader.sh
```

The script runs focused loader tests, decodes 20 deterministic manifest samples, verifies a `4x3x224x224` DataLoader batch, checks repeat access, and writes `aligned_samples.jpg` plus `hls_verification_report.json` under `$HOME/datasets/outputs/frodobots_2k_phase2/`. Inspect the contact sheet manually before treating image-label alignment as validated.

Before Phase 3, run the full semantic and edge-case audit on Dell:

```bash
./research/scripts/audit_phase2_alignment.sh
```

The audit builds a read-only manifest for all rides, creates early/middle/late temporal strips from at least five rides, prioritizes HLS-discontinuity rides, visualizes LEFT/RIGHT and available REVERSE samples, and reports transform, batch, monotonicity, and decode-failure details. Its automatic result remains `CONDITIONAL PASS` until `left_strips.jpg` and `right_strips.jpg` are reviewed by a person.

## Phase 3 ResNet18 Tiny Overfit

After Phase 2 passes, run the 200-sample GPU overfit gate on Dell:

```bash
./research/scripts/verify_phase3_tiny_overfit.sh
```

The script selects 40 samples from each action class, caches only those decoded tensors in memory, fine-tunes an ImageNet-pretrained ResNet18, and verifies loss reduction, at least 95% training accuracy, checkpoint reload, deterministic inference, and raw-dataset immutability. The first run may download the torchvision ResNet18 weights. Outputs remain outside Git under `$HOME/datasets/outputs/frodobots_2k_phase3/tiny_overfit/`.

After the tiny-overfit gate passes, run the bounded 10/2/2 ride-level baseline:

```bash
./research/scripts/run_phase3_small_baseline.sh
```

The run uses at most 250 samples per ride, selects the best checkpoint by validation macro F1, evaluates the held-out test rides once, and writes `held_out_test_predictions.mp4` with ground truth, prediction, confidence, and control overlays. Metrics, the exact ride split, class distributions, and confusion matrices are stored in `small_baseline_report.json` under `$HOME/datasets/outputs/frodobots_2k_phase3/small_baseline/`.

## Traversability temporal review

After the approved SegFormer-B0 static-image baseline, run `research/scripts/run_traversability_temporal_inference_v1.sh` on Dell. It selects three deterministic 30-second segments from rides absent from the complete approved 120-image split, then writes raw prediction/confidence videos, per-frame statistics, anomaly candidates, latency/FPS/VRAM metrics, and a portable `review.html` bundle. It does not smooth predictions, threshold confidence, train the model, or connect to rover control.

After human temporal review, `research/scripts/build_traversability_hard_examples_v1.sh` mines only the reviewed unseen rides for OFF_ROAD transitions and high-confidence predictions. It rechecks selected frames with the frozen v1 checkpoint, suppresses temporal and perceptual duplicates, isolates hard-train and hard-validation by ride, and targets 24 CVAT images: 12 curb hard negatives, 6 true OFF_ROAD scenes, and 6 paved hard cases. Category names are unverified sampling suggestions; every seed mask requires human correction before any v2 training. If a category is short, the script records the exact shortfall and builds only the relevant available subset without unrelated backfill.

For broad manual v2 review, run `research/scripts/build_manual_candidates_v2.sh` on Dell. It performs a 12-image dry run, then deterministically extracts about 200 untouched `uid_s_1000` front-camera frames from `output_rides_1` and `output_rides_2`. Approved v1 and prior hard-example rides are excluded, selection favors one image per ride with at most two separated by 20 seconds, and numbered contact sheets contain 25 candidates each. No model inference or training is involved.

After contact-sheet review, `research/scripts/select_manual_candidates_v2.sh` validates the approved candidate numbers and creates a separate 33-image CVAT upload bundle without changing the original JPG bytes or source review bundle.

After CVAT export, `research/scripts/import_validate_manual_traversability_v2_33.sh` imports only `SegmentationClass` masks into a separate validated 33-image bundle, verifies the existing v1 label contract and original JPG bytes, and creates overlay contact sheets in groups of at most 25. It does not merge with approved v1 data or start training.

Build the approved 153-image v2 dataset on Dell with `research/scripts/build_traversability_dataset_v2.sh`. It preserves the existing v1 validation/test assignments, adds only group-isolated manual samples to training, and reserves new source-ride groups as `new_holdout`. The builder rejects label-contract differences, exact image duplicates, ride overlap, and existing output paths.

After inspecting the v2 split report, run `research/scripts/run_traversability_segformer_b0_v2.sh` explicitly on Dell. It initializes from the approved v1 best checkpoint, uses the lower learning rate in `research/configs/traversability_segformer_b0_v2.yaml`, and compares v1/v2 on the fixed v1 validation/test sets and the new holdout. A separate from-ADE baseline is optional and must use a different output directory by running `research/training/train_traversability_segformer.py` without `--initial-checkpoint`; it is not part of the default v2 workflow.

After the v2 static and temporal overlays pass human review, run the first
log-only planner integration with
`research/scripts/run_traversability_planner_replay_v2.sh`. The script reuses the v2
checkpoint loader, aspect-preserving preprocessing, HLS lazy decoder, H.264
writer, Urban controller, safety monitor, and command filter. Raw FrodoBots
recordings have no mission waypoint, so `GOAL_HEADING_ERROR_DEG` is an explicit
fixed replay input rather than recorded GPS. Every output row records
`command_transmitted=false`; this workflow does not import or call the SDK.

## Traversability Pseudo-Label Pilot

The action baseline remains unchanged. Traversability pseudo-labeling is a separate research-only workflow using `nvidia/segformer-b0-finetuned-ade-512-512` as an annotation draft, never as verified ground truth or a rover controller.

On Dell, install the optional pinned dependency and preserve its resolver report:

```bash
./research/scripts/setup_traversability_pilot.sh
```

Then build the 40-frame, eight-ride default review bundle:

```bash
./research/scripts/run_traversability_pilot.sh
```

Paths can be overridden with `DATASET_ROOT`, `MANIFEST_PATH`, `BUNDLE_ROOT`, `SAMPLE_COUNT`, `MAX_RIDES`, `MINIMUM_SEPARATION_SECONDS`, and `SEED`. The default bundle is `$HOME/datasets/review_bundles/traversability_pilot_v1/`. Open `gallery.html` on the Mac and edit reviewer columns in `review.csv`. Do not train until the reviewed CSV and any corrected masks pass `research/training/validate_traversability_review.py` and the user explicitly approves them.

Copy the self-contained bundle by running this on the Mac, replacing both placeholders:

```bash
rsync -ah --progress \
  <DELL_USER>@<DELL_TAILSCALE_IP>:~/datasets/review_bundles/traversability_pilot_v1/ \
  <MAC_DESTINATION>/traversability_pilot_v1/
```

## Traversability Dataset v1 Annotation Pilot

After reviewing the pseudo-label pilot, prepare exactly 20 images for manual four-class annotation without rerunning SegFormer:

```bash
./research/scripts/run_traversability_annotation_pilot.sh
```

The default output is `$HOME/datasets/generated/traversability_dataset_v1/pilot_20/`. Follow `docs/training/traversability_dataset_v1_annotation.md` for the CVAT workflow and Dell-only import/validation commands. The required IDs are `0 IGNORE`, `1 ON_ROAD`, `2 OFF_ROAD`, and `3 OBSTACLE`. Do not expand the dataset or train from these masks until the user reviews and approves the completed pilot.

After explicit approval of the imported 20-image pilot, build the additional 100-image CVAT bundle on Dell:

```bash
./research/scripts/build_traversability_annotation_100.sh
```

This performs bounded pseudo inference on 240 manifest samples for selection and seed drafts, not full-dataset inference. It excludes the approved pilot by provenance, time, and visual hash; caps each ride at five selected images; and writes the new bundle outside Git under `$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1/`. Stop after generation and annotate all 100 images manually before any training.

After exporting the completed 100-image task from CVAT, validate it on Dell without touching the approved pilot:

```bash
./research/scripts/import_validate_traversability_annotation_100.sh
```

The reviewed output remains separate under `annotation_100_v1/reviewed_import/`. A validator PASS is not permission to merge the 20- and 100-image sets or start fine-tuning; inspect the generated overlays first.

After explicit human approval of both reviewed bundles, run the gated 120-image SegFormer baseline on Dell:

```bash
./research/scripts/run_traversability_segformer_b0_v1.sh
```

The script rebuilds and validates `$HOME/datasets/generated/traversability_dataset_v1/approved_120_v1/` from approved normalized masks only, creates a deterministic ride-level split, and runs focused loader tests. A six-image CUDA overfit gate must pass before the unweighted full fine-tuning starts. Training artifacts are written outside Git under `$HOME/datasets/experiments/traversability_segformer_b0_v1/`; test metrics are evaluated once from the best validation checkpoint, and the portable offline overlays are in `full_training/review_bundle/`. This workflow does not integrate with the SDK, planner, controller, or live rover.

## Berkeley-FrodoBots-7K Probe

Do not download the full Berkeley-FrodoBots-7K dataset during initial work. It is too large for local iteration.

## First Probe

Install optional training dependencies:

```bash
.venv/bin/pip install datasets huggingface_hub
```

Authenticate once for the gated dataset:

```bash
huggingface-cli login
```

Stream a small sample:

```bash
.venv/bin/python research/training/explore_berkeley_frodobots_7k.py --max-rows 200
```

Outputs:

- `datasets/berkeley_7k_probe/summary.json`
- `datasets/berkeley_7k_probe/sample_rows.jsonl`
- `datasets/berkeley_7k_probe/parsed_actions.csv`

## Decision Gate

After the probe, inspect:

- which action key is present: `action_mbra`, `action`, or `action_original`
- whether actions are exposed as numeric arrays or encoded payloads
- whether `__url__` groups samples by shard or source file
- whether image paths are available directly or require video extraction

Only after this should we build the real PyTorch `Dataset`.

## SAM-TP Phase 1 Video Gate

Run `research/scripts/run_sam_tp_phase1_video_review.sh` on Dell to create a
QuickTime-compatible deterministic review video. A display-only path starts
near the bottom of the image and is drawn only when a connected corridor stays
above the configured SAM-TP score threshold. The fourth panel separately shows
the seven fixed rover-frame candidates. The image-space path is not a metric
trajectory and must not be converted into steering before camera calibration.

The read-only SDK shadow path applies the same SAM-TP output adapter to live
front-camera frames, generates the same image-space proposal, and records the
same candidate count. Neither path calls an SDK write endpoint or transmits a
rover command.
