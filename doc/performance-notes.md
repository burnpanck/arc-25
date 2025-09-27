
# Training performance

## 1. round of experiments; no attention dropout, no mixed precision, no lax.scan

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
