# SAM2 HF backend config (vendored, no runtime network access)

These files pin the exact architecture/preprocessing needed to load
`best_sam_tp.pt` through `transformers.Sam2Model` (image-only SAM2, no
video-memory weights). They are consumed by
`training/sam_tp_hf_backend.py` and must not require network access at
runtime.

## Provenance

- `sam2_config.json`: generated from `Sam2Config().to_dict()` in
  `transformers==5.4.0` (the project-pinned version). `Sam2Config()`'s
  zero-argument defaults for `vision_config` / `prompt_encoder_config` /
  `mask_decoder_config` were cross-checked field-by-field against the
  published `facebook/sam2.1-hiera-tiny` Hub config
  (revision `de431c4043854a71d8101e17995dfe596bf101a5`,
  `config.json` sha256 `860aff9751b139d83a4ad7df1e5535416fded533e0ead02625edbefcb9953cce`)
  and are identical for every field that both configs define (backbone
  stage/embed/head dims, global attention blocks, window sizes, mask
  decoder head dims, prompt encoder point/mask embedding sizes, image
  size 1024). The Hub config is `Sam2VideoModel` (adds memory
  attention/encoder fields `Sam2Config`/`Sam2Model` do not have); those
  extra fields are not present in `best_sam_tp.pt` either, consistent
  with it being an image-only `Sam2Model` export.
  Empirically confirmed: instantiating `Sam2Model(Sam2Config(**this json))`
  and loading `best_sam_tp.pt` with `strict=True` succeeds with 0 missing,
  0 unexpected, and 0 shape-mismatched keys across all 309 tensors.
- `image_processor_config.json`: the image-only subset (resize/normalize
  fields) of the same Hub revision's `preprocessor_config.json`
  (sha256 `6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d`).
  `mask_size`/`processor_class`/`image_processor_type` (video/mask-map-only
  fields) were dropped since this backend only does point-prompted image
  segmentation. Values match GENIE's own `SAM2Transforms` (same 1024x1024
  bilinear resize, same ImageNet mean/std), so both checkpoints' backends
  preprocess images identically.

Regenerate `sam2_config.json` (if the pinned `transformers` version ever
changes) with the sam_tp_repro venv:

```
python -c "
import json
from transformers import Sam2Config
json.dump(Sam2Config().to_dict(), open('sam2_config.json', 'w'), indent=2, sort_keys=True)
"
```

then re-verify against `best_sam_tp.pt` with a strict `load_state_dict`
before trusting it.
