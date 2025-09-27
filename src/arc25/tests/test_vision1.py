import numpy as np
from flax import nnx

from ..vision1.classification import ARCClassifier
from ..vision1.fields import FieldDims


def test_classification():
    dims = FieldDims.make(
        inv_fac=2,
        context=32,
        hdrs=16,
        cells=16,
    )
    arc_cls = ARCClassifier(
        hidden_size=dims,
        mha_features=24 * 2,
        mlp_width_factor=2,
        num_heads=2,
        num_groups=1,
        num_classes=400,
        num_layers=8,
        rngs=nnx.Rngs(0),
    )
    arc_cls(np.zeros((1, 8, 8), int), np.array([[8, 8]]))
