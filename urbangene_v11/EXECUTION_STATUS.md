# V1.1 Execution Status

Updated 2026-11-09 00:50 UTC.

## Current Results Snapshot

- The activeguard queue is training on GPU 0, with 3 of 14 runs completed.
  R0, R2 and GRCT seed 42 each finished 30 epochs / 18,750 steps; R1 alpha
  is currently training. Formal test and A+B-to-C evaluation have not run.
- The frozen AE passes the full validation gate: PSNR 30.674 dB.
- Best latent L1 over the full 2,000-scene validation set: R0 0.8034
  (epoch 30), R2 0.8404 (epoch 1), GRCT 0.9154 (epoch 1). Lower is better.
  GRCT is 13.9% worse than R0 on this metric and deteriorates to 1.0727
  by epoch 30. GRCT gene usage falls from 24 to 4 types and mean latent
  spatial coverage from 42.67% to 11.98% between epochs 1 and 30.
- CPU inspection of the first 64 validation scenes using the selected best
  checkpoints gives RGB PSNR: AE 30.65, R0 14.37, R2 14.63, GRCT 13.65 dB.
  This subset diagnostic is separate from the full validation latent scores.
  R0 emits exactly one identical serialized program for all 64 inputs.
- Current outputs are synthetic street scenes, not real street-view photos.
  The Gene reconstructions fail to retain input scene structure; current
  evidence does not support a reconstruction or compositional quality gain.
- The visual comparison, metrics and Chinese report are saved at:
  `/host/mnt/nas/urban-visual-gene/urbangene_v11_20261108/v1_grct/preliminary_validation_20261109_0049/RESULTS.md`.
  Diagnostics only read completed checkpoints and use val_seen, leaving the
  running training source, queue, checkpoints and test splits untouched.
- Follow-up inspection found that R0's parser categorical outputs have mean
  max probability >99.99% on the first 32 validation scenes, and late-step
  gene/active/scale head gradients are zero. GRCT's last-checkpoint coverage is
  12.19% on the same 32 scenes; 94.43% of their latent L1 comes from uncovered
  positions, which are forced to zero by the representation. Details and
  caveats are in the NAS `DIAGNOSIS.md`.

## Completed

- GRCT, fixed controls, parameter-matched controls, ablations, geometry, masking,
  permutation checks and target-free recombination are implemented.
- V1.1 unit tests after the recovery fix: 23 passed. Original V0 regression
  tests: 16 passed.
- CPU smoke ran an 8-epoch reduced AE and 12 epochs each for R0, R2 and GRCT,
  using the same smoke AE and 272-bit program format.
- Smoke reconstructions and aligned seen/heldout A+B-to-C metrics were written
  to the NAS. The GRCT checkpoint selected by smoke validation has test-seen
  latent L1 0.2635, RGB L1 0.1691 and PSNR 13.92 dB. Its small AE baseline is
  15.33 dB. All eight test-seen images exceed the 0.1 RGB-L1 failure threshold.
  These results confirm execution only; the reduced smoke configuration is not
  a quality claim.
- The smoke recombination run retained all eight samples per split, with no
  overflow. GRCT full-image RGB L1 was 0.1614 (seen) and 0.1738 (heldout).
- The revised CPU smoke uses a deterministic non-empty fallback when the parser
  would otherwise close every slot. GRCT finished with a mean of 5.5 active
  instances, test-seen latent L1 0.2647, RGB L1 0.1679, and PSNR 14.01 dB;
  all 8 samples still exceed the smoke RGB-L1 failure threshold. Its paired
  RGB L1 was 0.1603 (seen) and 0.1720 (heldout), with no overflow. This remains
  a pipeline diagnostic, not evidence of acceptable output quality.
- V0 source and original queue/checkpoint locations were not edited.

## Formal Run

The full plan contains 14 runs: R0/R2/GRCT seed 42, fixed alpha and GRCT
ablations, parameter-matched R0-P/R2-P, then core seeds 29 and 43. All use the
same separately trained AE. The queue snapshots source/config hashes, waits for
three stable GPU availability observations, and enforces an AE validation PSNR
gate of 24 dB before training the Gene models. A gate failure is reported and
halts the run rather than attributing AE loss to the composer.

The first formal queue at
`outputs/v1_grct/formal_v11_20261108/` reached the AE gate (30.67 dB) and
completed seed-42 R0 and R2. GRCT then collapsed to an all-inactive program;
after that, all monitored GRCT gradients were zero. That queue was stopped
after epoch 2 of GRCT. Its files were retained as diagnostic output and are not
included as completed results.

The implementation now removes the redundant hard coverage multiplier after
the R0 base and masked spatial renderer, preserving the ST recovery gradient
while retaining exactly zero forward values outside coverage. V1.1 also
guarantees at least one active instance only when all 16 parsed slots would be
inactive; the fallback is chosen from serialized Gene/position/scale values.
This keeps 272 bits and adds no loss. The revised CPU smoke avoids empty scenes,
with 5.5 active instances on average in the selected checkpoint, so its quality remains
insufficient and requires scrutiny during formal validation.

The revised implementation uses the separate NAS plan directory
`outputs/v1_grct/formal_v11_20261108_activeguard/` and reuses a byte-identical
copy of the already-qualified frozen AE. Formal model weights started fresh.
Its queue is running (PID 399150) on GPU 0. See the current results snapshot
above for the latest training and validation state.
