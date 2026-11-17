# V2 M1 Protocol

Current mainline: real Google Street View photographs. See
`REAL_STREET_PROTOCOL_20261109.md` for the 512/64 panorama-separated RGB
experiment, fresh DINOv3 extraction, and label-free evaluation limits.
The numbered protocol below describes the archived synthetic diagnostic
screens. Synthetic evidence does not establish real-image object discovery.

1. Record DINOv3 model ID/revision, full checkpoint/config hashes, input
   resize/crop/normalization, patch grid, feature layer, and cache manifest.
2. Generate disjoint synthetic splits from the existing V0 rule and seed.
   Store image-only manifests separately from evaluation instance masks.
3. Freeze the pretrained DINOv3 encoder. Train Adaptive Spatial Slot
   Discovery to reconstruct its dense feature map; the loader receives only
   feature tensors. No class, instance mask, box, or count is passed to it.
4. Select checkpoints by validation feature reconstruction only. Use synthetic
   instance masks only in the final discovery evaluator.
5. Report Hungarian matched mask IoU, foreground ARI, count MAE, split/merge
   fractions, feature reconstruction MSE, and feature-map flip stability.
   Compare against one-slot, regular-grid, and random-grouping diagnostics.
6. Keep `test_seen`, `test_unseen_combo`, and `test_higher_complexity` locked
   until validation diagnostics and a configuration hash are recorded.

The initial CPU smoke on the existing real-image feature cache is an
engineering check only. It does not satisfy the synthetic instance-discovery
gate. A scientific Gate decision requires the extracted synthetic validation
features, matched instance metrics, and saved failure visualizations.
