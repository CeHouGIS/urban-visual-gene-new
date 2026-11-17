# V1.1 implementation audit

2026-11-08. Supplied archived filename contains the V1.1 GRCT specification; that content is the implementation contract (copied verbatim to PRD_V1.1.md).

## Existing implementation and isolation

- V0 package: ../urbangene/src/urbangene. Existing parser produces integer gene_ids/enabled/position_ids/scale_ids plus hard-valued ST surrogates. Reuse parser, AE, shared GeneLibrary, metrics, data rules and 272-bit wire codec by import.
- V0 outputs/v0 queue is waiting for resources. No formal AE checkpoint exists. V0 smoke checkpoint has 4 channels and is unsuitable for the formal 8-channel interface.
- V0 hashes all Python source recursively. V1.1 therefore lives in a separate package/repository directory; no edits to V0 source, configs, queue or checkpoints.
- System disk reports zero available capacity. V1.1 outputs is a symlink to a dedicated directory on the second workspace filesystem (approximately 8.6 TiB available at audit). No old files are deleted or migrated.
- If a formal AE is unavailable, V1.1 prepares an independent AE using the unchanged V0 architecture/configuration, under its own outputs. Every V1.1 comparison loads that SAME checkpoint.

## New modules

Implement geometry sampling, spatial template embedding, relational attention, explicit pixel-instance values, GRCT residual rendering, fixed alpha and residual CNN controls; no changes to scene-specific wire fields.

Implement deterministic aligned A/B generation and a separate scoring-only target API; merge uses serialized source programs and predefined region/overlap threshold. Overflows are retained as failures, never silently truncated.

## Gradient and masking risks

- Integer indexing alone cuts parser gradients. ST surrogates must have exactly the same forward values as the serialized program, including position/scale/active.
- A zero output projection blocks interior Transformer gradients on the first update. Tests must check initialization equality and verify interior gradients AFTER the output head has received an update, rather than treating expected first-step zeros as permanent collapse.
- All-invalid attention must explicitly return zero, not softmax(-inf). Inactive instances cannot influence other instances; uncovered canvas positions are zero.
- No learned slot-index embeddings. Main GRCT must be permutation invariant; relation output equivariant. Alpha control is deliberately order dependent.
- A+B inference has no target argument. Target C is constructed/read only in a scoring method, and never parsed to choose slots.

## Milestones

M0 audit → M1 architecture/invariant tests → M2 CPU smoke/overfit diagnostic → M3 shared-AE R0/R2/R3 comparison → M4 locked reconstruction/recombination evaluation and seed confirmation. Execution reports must distinguish implemented, executed, running and pending stages.
