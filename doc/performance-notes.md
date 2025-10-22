
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


### MAE 1

Compilation time on TPU: ~6 min
Measurements with a single bucket at 30x30:

- *TPU v5e-8*: global batch size 128, remat=True, bf16, mode="flat": OOM
- *TPU v5e-8*: global batch size 96, remat=True, bf16, mode="flat": ~79 images/s
- *TPU v5e-8*: global batch size 64, remat=True, bf16, mode="flat": ~69 images/s


### MAE 2; `tiny` config

Batch size calculated using a reference size of 15, base cost of 10.
Observed compilation time on 4x L4: 750 seconds (somehow, we got four traces).

- *4x L4*: Batchsize scaling  256, only 30x30, bf16, mode="flat":  ~73 wps
- *4x L4*: Batchsize scaling  512, only 30x30, bf16, mode="flat":  ~83 wps
- *4x L4*: Batchsize scaling 1024, only 30x30, bf16, mode="flat": OOM
- *4x L4*: Batchsize scaling  768, only 30x30, bf16, mode="flat": OOM
- *4x L4*: Batchsize scaling  512, only 12x12, bf16, mode="flat": ~170 wps


## Vertex AI Batch size tuning

### Model config `small` on single L4
mode="flat", bf16.

Cost: $1.05/h/GPU on-demand


- Image size 30x30:
  - 1: 0.51 wt/s 0.51
  - 2: 0.96 wt/s 0.94
  - 4: 1.53 wt/s 1.55
  - 5: 1.63 wt/s 1.65
  - 6,8: OOM

- Image size 24x24:
  - 4: 1.82 wt/s 2.85
  - 6: 2.50 wt/s
  - 7: 2.42 wt/s
  - 8: OOM

- Image size 20x20:
  - 8: 3.36 wt/s
  - 9, 10, 12, 16: OOM

- Image size 16x16:
  -  8: 3.53 wt/s
  - 10: 4.64 wt/s
  - 11: OOM

- Image size 12x12:
  -  8: 4.03 wt/s
  - 12: 5.96 wt/s
  - 13, 14, 16: OOM

- Take 2: Image size 12x12:
  -  1: 0.51 wt/s
  -  2: 0.96 wt/s
  -  4: 1.94 wt/s
  -  8: 3.94 wt/s
  - 16: 7.18 wt/s
  - 32:10.33 wt/s
  - 34 ... 64: OOM

- Take 3 (maxmem): Image size 12x12:
  -  1: 0.51 wt/s
  -  2: 0.97 wt/s
  -  4: 1.99 wt/s
  -  8: 3.93 wt/s
  - 16: 7.19 wt/s
  - 32:10.32 wt/s
  - 40:11.17 wt/s
  - 48, 64: OOM

- Test-run: Image size 12x12:
  - 16:  7.1 wt/s
  - 32: 10.3 wt/s
  - 33 ... 64: OOM

### Model config `small` on single TPU v6e
mode="flat", bf16.

Cost: $2.97/h/TPU on-demand (spot unavailable!)

- Image size 30x30:
  -  1: 0.81 wt/s
  -  2: 1.54 wt/s
  -  4: 3.17 wt/s
  -  8: 4.36 wt/s
  - 12: 5.09 wt/s
  - 14: 5.46 wt/s
  - 15: OOM
  - 16: OOM
- Image size 24x24:
  - 16: 7.76 wt/s
  - 20: 8.07 wt/s
  - 22: 7.60 wt/s
  - 23: 7.76 wt/s
  - 24, 32: OOM
- Image size 20x20:
  - 16: 8.02 wt/s
  - 24: 9.44 wt/s
  - 28: 8.83 wt/s
  - 29: 9.38 wt/s
  - 30, 32: OOM
- Image size 16x16:
  - 32: 12.84 wt/s
  - 48: 14.03 wt/s
  - 50, 52, 56, 64: OOM
- Image size 12x12:
  - 64: 19.72 wt/s
  - 80: 19.83 wt/s
  - 84, 88, 96, 128: OOM

- Take 2 (maxmem): Image size 12x12:
  -  1:  0.78 wt/s
  -  2:  1.47 wt/s
  -  4:  3.03 wt/s
  -  8:  6.05 wt/s
  - 16: 10.44 wt/s
  - 32: 15.19 wt/s
  - 64: 19.67 wt/s
  - 80: 19.53 wt/s
  - 81...: OOM


### H100

Cost: $14.07/h/GPU on-demand
