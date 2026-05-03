# Cross-Attention Tight Fusion (Vision + IMU)

## Overview

Tight fusion of [BranchA](../vision_only/model.py) (vision) and
[AirIONet](../imu_only/model.py) (IMU) features through a transformer
that performs cross-modal attention. Each modality contributes one
token per timestep (after a linear projection to a shared dimension);
the two streams are concatenated along the sequence axis with a
learned modality embedding so the transformer can distinguish them,
and several encoder blocks of multi-head self-attention let every
token reason about every other token.

This is the heaviest of the three fusion approaches — a few million
extra parameters versus the gated variant — but in exchange the
network learns *per-token* cross-modal interactions: a vision token at
time `t` can attend to an IMU sample at any time, and vice versa.

## Pipeline

```
        frames_t, frames_t1                 imu_acc, imu_gyro, attitude
                │                                       │
                ▼                                       ▼
            BranchA                                 AirIONet
       (frozen pretrained)                      (frozen pretrained)
                │                                       │
            features [B,T,128]                   bi-GRU output [B,T*W,256]
                │                          take last sample of every window
                │                                       │
                ▼                                       ▼
           vis_proj                                  imu_proj
                │ + pos + modality_emb[0]                │ + pos + modality_emb[1]
                └────────────────┬──────────────────────┘
                                 ▼
                       concat tokens   [B, 2T, D]
                                 ▼
                ┌───────────────────────────────────┐
                │ Transformer encoder blocks        │
                │ (self-attention over 2T tokens)   │
                │ ≡ vision↔IMU cross-attention      │
                └────────────────┬──────────────────┘
                                 ▼
                  read out vision + IMU per t,
                  average them → fused [B, T, D]
                                 ▼
                          LayerNorm + heads
                       ┌─────────┴─────────┐
                       ▼                   ▼
                   trans (3)          rot_6d (6)
                                          │
                                    Gram-Schmidt
                                          │
                                       R (3x3)
```

## Why concatenated self-attention is cross-attention

If you flatten the two streams into a single sequence of `2T` tokens
and compute self-attention over them, every token's output is a
weighted combination of *all* tokens — including tokens from the other
modality. This is mathematically equivalent to alternating layers of
within-modality self-attention and across-modality cross-attention,
but it shares parameters and is much simpler to implement. The
modality embedding tells the network which is which so the attention
heads can specialize.

## Training

```bash
python train.py \
    --data_root /path/to/dataset \
    --vision_checkpoint ../vision_only/checkpoints/branch_a/best.pt \
    --airio_checkpoint  ../imu_only/checkpoints/airio/best.pt
```

| Hyperparameter | Default | Notes |
|---|---|---|
| feat_dim    | 128 | shared model dim |
| num_heads   | 4   | multi-head attention heads |
| num_layers  | 2   | transformer encoder blocks |
| ffn_hidden  | 256 | FFN width inside each block |
| dropout     | 0.1 | inside transformer layers |
| optimizer   | Adam, weight decay 1e-4 |
| lr (warm-up)| 1e-4 | new layers only |
| lr (joint)  | 2e-5 | unfreezes both backbones |
| warmup_epochs | 5 | epochs of frozen-backbone training |
| epochs      | 100 | total |
| batch size  | 8   | vision images dominate memory |

## Evaluation

```bash
python evaluate.py --data_root /path/to/dataset \
                   --checkpoint checkpoints/cross_attention/best.pt
```

Same metrics as the other branches (ATE, RTE-5s, mean rotation deg).
Per-sequence plots include a **vision-IMU cosine similarity** plot —
the closest direct analogue to the gated branch's gate plot. Values
near 1 mean the modalities agree; values near 0 mean the transformer
is reconciling disagreement.

## When to prefer this over gated fusion

- Real-world datasets with diverse failure modes (motion blur, dynamic
  scenes, IMU vibration spikes).
- Long-context modelling where temporal cross-modal lookups matter
  (e.g. "this frame is blurry, but the IMU two steps ago shows I'm
  about to stop spinning").
- Sufficient training data (synthetic-only data with limited variation
  often *under-uses* the transformer's capacity).

For shorter, cleaner synthetic Blender data, the gated variant
typically reaches comparable accuracy with one-tenth the parameters
and trains 2-4× faster.

## References

- Vaswani et al., 2017. "Attention Is All You Need." *NeurIPS*.
- Devlin et al., 2019. "BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding." *NAACL*.
- Tsai et al., 2019. "Multimodal Transformer for Unaligned Multimodal
  Language Sequences." *ACL*.
- Clark et al., 2017. "VINet." *AAAI*.
