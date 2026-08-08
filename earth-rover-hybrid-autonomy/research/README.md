# research/

Everything under here is *not* part of the live rover pipeline. It's the old
offline ML training/dataset-building work (SegFormer traversability model,
FrodoBots-2K/Berkeley dataset exploration, CVAT annotation import, action
classifier baselines) plus a couple of resolved one-off diagnostics.

Nothing in this tree is imported by, or required to run, the live rover
scripts at the top level (`scripts/run_sam_tp_sdk_shadow.sh`,
`scripts/run_mission1_autonomy.sh`, `scripts/teleop_dashboard.py`,
`scripts/view_sdk_camera.py`, `scripts/run_sdk_smoke_test.py`,
`scripts/compare_mission1_planners.py`) or by the `earth_rover` library in
`src/`. It mirrors the original flat layout (`training/`, `scripts/`,
`configs/`, `tests/`) one level deeper, under this directory.

See `research/training/README.md` for the offline training workflow itself.

`pyproject.toml`'s `testpaths` only points at the top-level `tests/`, so a
bare `pytest` run does not exercise anything in here — run
`pytest research/tests` explicitly if you're working in this tree.
