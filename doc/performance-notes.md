
# Training performance

## Vision 1

All of the following performance data use the following base configuration (*"nano*"):
```python
dims = FieldDims.make(
    inv_fac = 2,
    context = 32,
    hdrs = 16,
    cells = 16,
)
arc_cls = ARCClassifier(
    hidden_size = dims,
    mha_features = 24*2,
    mlp_width_factor = 2,
    num_heads = 2,
    num_groups = 1,
    num_classes = 400,
    num_layers = 8,
    rngs = nnx.Rngs(0),
)
config = TrainConfig(
    global_batch_size = 16,
    num_train_steps = 1000,
    warmup_steps = 5,
    learning_rate = 3e-4,
    weight_decay = 0.05,
)
```

Furthermore, the numbers assume very little bucketing, only two buckets [16, 30].

Training performance:
- *2x T4*: global batch size 8, ~14 images/s
- *2x T4*: global batch size 16, ~16 images/s
- *2x T4*: global batch size 32, OOM
- *4x L4*: global batch size 128, OOM
- *4x L4*: global batch size 64, OOM
- *4x L4*: global batch size 32, ~55 images/s
- *4x L4*: global batch size 16, ~46 images/s
- *Apple M1 CPU*: 2 virtual devices, global batch size 16: unroll=False & remat=True: 1.3 images/s
- *Apple M1 CPU*: 2 virtual devices, global batch size 16: unroll=True & remat=False: 1.2 images/s
- *Apple M1 CPU*: 2 virtual devices, global batch size 16: unroll=False & remat=False: 1.07 images/s
- *4x L4*: global batch size 128, remat=True, fp32: OOM
- *4x L4*: global batch size 64, remat=True, fp32: ~ 50 images/s
- *4x L4*: global batch size 128, remat=True, bf16/mixed: OOM :-(
- *4x L4*: global batch size 64, remat=True, bf16/mixed: ~ 70 images/s

## Vision 2

### Configuration 0

```python
width = FieldDims(
    context = SymDecompDims(
        space = 2*16,   # 8x2 = 16
        flavour = 1*16, # 10x1 = 10
        invariant= 14*16, # 1x14 -> 40*16
    ),
    cells = SymDecompDims(
        space = 2*8,
        flavour = 1*8,
        invariant = 22*8, # -> 48*8
    ),
    context_tokens = 1,
)

arc_cls = ARCClassifier(
    num_classes = 400,
    num_heads=8,
    num_groups=2,
    num_layers=8,
    hidden_size=width,
    swiglu_width_factor=8/3,
    qk_head_width=SymDecompDims(
        space = 3 * 8,  # 1x3x8
        flavour = 1 * 4,  # 10x1x4 = 5x1x8
        invariant = 4 * 8, # 1*4*8 -> 12x8
        rep=RepSpec(symmetry.ChiralityRep, 10)
    ),
    v_head_width=SymDecompDims(
        space = 2*4,
        flavour = 1*4,
        invariant = 14*4,
    ),
    use_chirality_rep=False,
#    kernel_init=quant,
#    bias_init=quant,
    per_head_rope_freq=False,
    dtype=jnp.bfloat16,
#    activation=jax.nn.relu,
#    use_bias=False,
    rngs=nnx.Rngs(42),
)
```
Compilation time on 4x L4: 2min 10s to 2min 40s
Measurements with a single bucket at 30x30:

- *4x L4*: global batch size 256, remat=True, fp32: OOM
- *4x L4*: global batch size 128, remat=True, fp32: OOM
- *4x L4*: global batch size 64, remat=True, fp32: ~ 46.3 images/s
- *4x L4*: global batch size 64, remat=True, fp32: ~ 133 images/s (at 15x15!)
- *4x L4*: global batch size 256, remat=True, fp32: ~ 198 images/s (at 15x15!)
- *4x L4*: global batch size 112, remat=True, bf16: OOM
- *4x L4*: global batch size 96, remat=True, bf16: ~ 65.4 images/s
- *4x L4*: global batch size 64, remat=True, bf16: ~ 62.1 images/s
- *4x L4*: global batch size 64, remat=True, bf16: ~ 156 images/s (at 15x15!)
- *4x L4*: global batch size 256, remat=True, bf16: ~ 261 images/s (at 15x15!)
- *4x L4*: global batch size 384, remat=True, bf16: ~ 279 images/s (at 15x15!)
- *4x L4*: global batch size 96, remat=True, bf16, mode="split": ~ 65.7 images/s
- *4x L4*: global batch size 96, remat=True, bf16, mode="flat": ~ 85.6 images/s
- *4x L4*: global batch size 128, remat=True, bf16, mode="flat": ~ 92.0 images/s
- *4x L4*: global batch size 256, remat=True, bf16, mode="flat": OOM
- *4x L4*: global batch size 192, remat=True, bf16, mode="flat": OOM
- *4x L4*: global batch size 160, remat=True, bf16, mode="flat": OOM
- *4x L4*: global batch size 144, remat=True, bf16, mode="flat": ~90.0 images/s

### Configuration 1 (with perceiver, still classification)

```python
width = FieldDims(
    context = SymDecompDims(
        space = 2*16,   # 8x2 = 16
        flavour = 1*16, # 10x1 = 10
        invariant= 14*16, # 1x14 -> 40*16
    ),
    cells = SymDecompDims(
        space = 2*8,
        flavour = 1*8,
        invariant = 22*8, # -> 48*8
    ),
    context_tokens = 2,
)

arc_cls = ARCClassifier(
    num_classes = 400,
    num_heads=8,
    num_groups=2,
    num_layers=8,
    num_perceiver_layers=3,
    num_perceiver_tokens=8,
    hidden_size=width,
    swiglu_width_factor=8/3,
    qk_head_width=SymDecompDims(
        space = 3 * 8,  # 1x3x8
        flavour = 1 * 4,  # 10x1x4 = 5x1x8
        invariant = 4 * 8, # 1*4*8 -> 12x8
        rep=RepSpec(symmetry.ChiralityRep, 10)
    ),
    v_head_width=SymDecompDims(
        space = 2*4,
        flavour = 1*4,
        invariant = 14*4,
    ),
    use_chirality_rep=False,
    per_head_rope_freq=False,
    dtype=jnp.bfloat16,
    rngs=nnx.Rngs(42),
)
```

Compilation time on 4x L4: 3min 30s
Measurements with a single bucket at 30x30:

- *4x L4*: global batch size 128, remat=True, bf16, mode="flat": ~ 76.0 images/s
- *4x L4*: global batch size 128, remat=True, bf16, mode="split": OOM
- *4x L4*: global batch size 96, remat=True, bf16, mode="split": ~ 57 images/s
- *TPU v5e-8*: global batch size 128, remat=True, bf16, mode="flat": ~ 158 images/s
- *TPU v5e-8*: global batch size 192, remat=True, bf16, mode="flat": OOM
- *TPU v5e-8*: global batch size 160, remat=True, bf16, mode="flat": ~ 136 images/s
