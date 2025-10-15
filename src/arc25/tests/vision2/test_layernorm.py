import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.layernorm import SymDecompLayerNorm
from ...vision2.symrep import RepSpec, SplitSymDecomp, SymDecompDims
from .conftest import quant, verify_swap


@pytest.mark.parametrize("norm_per", ["all", "rep", "basis"])
@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_SymDecompLayerNorm_symmetry(norm_per, use_fast_variance):
    """Test that LayerNorm respects D4 symmetry operations."""
    # Create test dimensions
    space_rep = symmetry.FullRep
    feat = SymDecompDims(17, 11, 5, rep=RepSpec(space_rep, 10))

    rngs = nnx.Rngs(42)
    layer_norm = SymDecompLayerNorm(
        feat,
        rngs=rngs,
        norm_per=norm_per,
        use_fast_variance=use_fast_variance,
        scale_init=quant,
        bias_init=quant,
    )

    batch_shape = (3,)

    # Create test input
    inp = SplitSymDecomp.empty(feat, batch=batch_shape)
    inp = attrs.evolve(
        inp,
        **{
            k: dict(invariant=1, space=8, flavour=10)[k]
            * jax.random.randint(rngs.params(), v.shape, -3, 4)
            / 2
            for k, v in inp.representations.items()
        },
    )

    # Apply layer norm to get expected output
    out = layer_norm(inp)

    # Verify D4 symmetry: transform(layernorm(x)) == layernorm(transform(x))
    for op in D4:
        # Get transformation indices
        tfo = symmetry.transform_coeff_in_basis(op, space_rep)

        # Transform input: only affects space representation
        tinp = attrs.evolve(
            inp,
            space=jnp.take(inp.space, tfo, axis=-2),
        )

        assert feat.validate(tinp), feat.validation_problems(tinp)

        # Transform expected output: only affects space representation
        expected = attrs.evolve(
            out,
            space=jnp.take(out.space, tfo, axis=-2),
        )

        assert feat.validate(expected), feat.validation_problems(expected)

        # Apply layer norm to transformed input
        actual = layer_norm(tinp)

        # Verify outputs match
        for k, v in expected.representations.items():
            a = getattr(actual, k)
            assert np.allclose(a, v, rtol=1e-5, atol=1e-5), f"{op}: {k}"

        # Verify flavour permutation symmetry
        verify_swap(op, tinp, expected, rngs, layer_norm)


@pytest.mark.parametrize("norm_per", ["all", "rep", "basis"])
@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_SymDecompLayerNorm_flat_vs_split(norm_per, use_fast_variance):
    """Test that flat and split modes produce identical results."""
    # Create test dimensions
    space_rep = symmetry.FullRep
    feat = SymDecompDims(17, 11, 5, rep=RepSpec(space_rep, 10))

    rngs = nnx.Rngs(42)
    layer_norm = SymDecompLayerNorm(
        feat,
        rngs=rngs,
        norm_per=norm_per,
        use_fast_variance=use_fast_variance,
        scale_init=quant,
        bias_init=quant,
    )

    batch_shape = (3,)

    # Create test input
    inp = SplitSymDecomp.empty(feat, batch=batch_shape)
    inp = attrs.evolve(
        inp,
        **{
            k: dict(invariant=1, space=8, flavour=10)[k]
            * jax.random.randint(rngs.params(), v.shape, -3, 4)
            / 2
            for k, v in inp.representations.items()
        },
    )

    # Apply with split mode
    out_split = layer_norm(inp, mode="split")

    # Apply with flat mode
    out_flat = layer_norm(inp, mode="flat")

    # Verify both modes produce identical results
    for k, v_flat in out_flat.representations.items():
        v_split = getattr(out_split, k)
        assert np.allclose(
            v_split, v_flat, rtol=1e-5, atol=1e-5
        ), f"{k}: split and flat modes differ"


@pytest.mark.skip
@pytest.mark.parametrize("norm_per", ["all", "rep", "basis"])
@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_SymDecompLayerNorm_normalization(norm_per, use_fast_variance):
    """Test that LayerNorm actually normalizes each representation."""
    # Create test dimensions
    space_rep = symmetry.FullRep
    feat = SymDecompDims(17, 11, 5, rep=RepSpec(space_rep, 10))

    rngs = nnx.Rngs(42)
    # Use scale=1, bias=0 to test pure normalization
    layer_norm = SymDecompLayerNorm(
        feat,
        rngs=rngs,
        norm_per=norm_per,
        use_fast_variance=use_fast_variance,
        scale_init=nnx.initializers.ones,
        bias_init=nnx.initializers.zeros,
    )

    batch_shape = (5,)

    # Create test input with varied statistics
    inp = SplitSymDecomp.empty(feat, batch=batch_shape)
    inp = attrs.evolve(
        inp,
        invariant=jax.random.normal(rngs.params(), inp.invariant.shape) * 10 + 5,
        space=jax.random.normal(rngs.params(), inp.space.shape) * 20 - 10,
        flavour=jax.random.normal(rngs.params(), inp.flavour.shape) * 5 + 15,
    )

    # Apply layer norm
    out = layer_norm(inp)

    # Check that each representation is normalized (mean ≈ 0, var ≈ 1)
    # Invariant: normalize over last axis
    inv_mean = jnp.mean(out.invariant, axis=-1)
    inv_var = jnp.var(out.invariant, axis=-1)
    assert np.allclose(inv_mean, 0, atol=1e-5), f"invariant mean: {inv_mean}"
    assert np.allclose(inv_var, 1, atol=1e-4), f"invariant var: {inv_var}"

    # Space: normalize over last two axes
    space_mean = jnp.mean(out.space, axis=(-2, -1))
    space_var = jnp.var(out.space, axis=(-2, -1))
    assert np.allclose(space_mean, 0, atol=1e-5), f"space mean: {space_mean}"
    assert np.allclose(space_var, 1, atol=1e-4), f"space var: {space_var}"

    # Flavour: normalize over last two axes
    flavour_mean = jnp.mean(out.flavour, axis=(-2, -1))
    flavour_var = jnp.var(out.flavour, axis=(-2, -1))
    assert np.allclose(flavour_mean, 0, atol=1e-5), f"flavour mean: {flavour_mean}"
    assert np.allclose(flavour_var, 1, atol=1e-4), f"flavour var: {flavour_var}"


@pytest.mark.skip
@pytest.mark.parametrize("norm_per", ["all", "rep", "basis"])
@pytest.mark.parametrize("use_fast_variance", [False, True])
def test_SymDecompLayerNorm_scale_and_bias(norm_per, use_fast_variance):
    """Test that flat and split modes produce identical results."""
    # Create test dimensions
    space_rep = symmetry.FullRep
    feat = SymDecompDims(17, 11, 5, rep=RepSpec(space_rep, 10))

    rngs = nnx.Rngs(42)

    # Create layer norm with specific scale/bias
    layer_norm = SymDecompLayerNorm(
        feat,
        rngs=rngs,
        norm_per=norm_per,
        use_fast_variance=use_fast_variance,
        scale_init=lambda key, shape, dtype: jnp.full(shape, 2.0, dtype=dtype),
        bias_init=lambda key, shape, dtype: jnp.full(shape, 3.0, dtype=dtype),
    )

    batch_shape = (2,)

    # Create test input
    inp = SplitSymDecomp.empty(feat, batch=batch_shape)
    inp = attrs.evolve(
        inp,
        invariant=jax.random.normal(rngs.params(), inp.invariant.shape),
        space=jax.random.normal(rngs.params(), inp.space.shape),
        flavour=jax.random.normal(rngs.params(), inp.flavour.shape),
    )

    # Apply layer norm
    out = layer_norm(inp)

    # With scale=2 and bias=3, each normalized value should be: 2*normalized + 3
    # So mean should be ≈3, std should be ≈2

    # Check invariant
    inv_mean = jnp.mean(out.invariant, axis=-1)
    inv_std = jnp.std(out.invariant, axis=-1)
    assert np.allclose(inv_mean, 3.0, atol=0.1), f"invariant mean: {inv_mean}"
    assert np.allclose(inv_std, 2.0, atol=0.1), f"invariant std: {inv_std}"

    # Check space
    space_mean = jnp.mean(out.space, axis=(-2, -1))
    space_std = jnp.std(out.space, axis=(-2, -1))
    assert np.allclose(space_mean, 3.0, atol=0.1), f"space mean: {space_mean}"
    assert np.allclose(space_std, 2.0, atol=0.1), f"space std: {space_std}"

    # Check flavour
    flavour_mean = jnp.mean(out.flavour, axis=(-2, -1))
    flavour_std = jnp.std(out.flavour, axis=(-2, -1))
    assert np.allclose(flavour_mean, 3.0, atol=0.1), f"flavour mean: {flavour_mean}"
    assert np.allclose(flavour_std, 2.0, atol=0.1), f"flavour std: {flavour_std}"
