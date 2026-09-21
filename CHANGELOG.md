# TGRIP Changelog

- **v1.1.0** (September 2026)
  - Fixed the view transform loop, which never fed a layer's output into the next
    one. Only the last layer reached the output, so the released weights are a
    single-layer BEVFormer and `n_layers` is now `1`.
  - Removed three sources of parameters that never received a gradient: the
    discarded view transform layers, the unused EfficientViT stage, and the sparse
    encoder projection shortcuts. DDP no longer needs `find_unused_parameters`.
  - Added `scripts/migrate_checkpoint.py` to convert pre-v1.1.0 checkpoints, and
    `tests/test_unused_parameters.py` to guard against new dead branches.
  - Checkpoints are now published as GitHub release assets and default to CLIP-L/14.

- **v1.0.0** (March 2026)
  - Initial release.
