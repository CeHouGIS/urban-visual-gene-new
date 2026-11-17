# UrbanGene V1.1

V1.1 is an isolated experiment package. It imports V0's scene parser, 8-channel
autoencoder, gene library, fixed renderer, dataset generator and 272-bit wire
format without editing the original package. Experiment outputs live on NAS
through `outputs/`.

## Model

Each scene uses the V0 program: 16 slots, 64 shared genes, quantized position
and scale, and a fixed 272-bit representation. The parser is trained with the
frozen AE and shared gene templates using latent L1 only.

The main composer is GRCT: instance embeddings feed a relation transformer,
then 1,024 fixed spatial queries attend to local, placed gene-template samples.
The initial output head is zero initialized, so the initial composition matches
the V0 fixed average renderer. It has no image, target latent, parser features,
scene labels or target mask in its input. Pixel output is zero outside the hard
program's coverage. The output is permutation invariant to slot order.
If all 16 parser slots are off, V1.1 enables one fallback instance selected
from the serialized Gene, position and scale fields. This prevents empty-scene
gradient collapse without adding a loss or changing the 272-bit program.

Controls include the fixed V0 average renderer (R0), fixed alpha ordering (R1),
Residual CNN (R2), relation and local-spatial ablations, and R0-P/R2-P controls
whose trainable parameter counts match GRCT within 0.1% for the current config.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../urbangene/src \
  python run.py smoke --config configs/smoke.yaml --device cpu
```

The CPU smoke trains a reduced AE and R0/R2/GRCT on a tiny synthetic dataset,
then evaluates reconstruction and aligned A+B-to-C recombination. It verifies
the full data-to-checkpoint-to-metrics path; its scores are not formal results.

For the full run, make a new NAS plan, then start its resource-waiting queue:

```bash
python run.py plan --output outputs/v1_grct/formal_v11_YYYYMMDD
python run.py queue --plan outputs/v1_grct/formal_v11_YYYYMMDD/plan.json
```

The queue locks source and configuration hashes, waits for an available GPU,
trains one shared AE, checks its validation PSNR gate, and then runs the
predeclared comparison and seed schedule. Training checkpoints, logs, cached
latents, metrics, images and source snapshots are kept under `outputs/`.

## Evaluation protocol

Recombination receives only A and B. Its source programs are serialized,
round-tripped through the V0 wire codec, then merged using a fixed region and
geometry overlap rule. C is rendered only after prediction for scoring. Any
program overflow is retained as a failed sample; it is never truncated.
