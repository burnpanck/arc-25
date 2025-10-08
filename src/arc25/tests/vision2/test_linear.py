import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.linear import SymDecompLinear
from ...vision2.symrep import RepSpec, SplitSymDecomp, SymDecompDims
from .conftest import quant, verify_swap


@pytest.mark.parametrize(
    "extra_in_reps,extra_out_reps,out_space_rep",
    [
        pytest.param((symmetry.AxialDirRep,), (), symmetry.FullRep, id="extra_in"),
        pytest.param((), (symmetry.AxialDirRep,), symmetry.FullRep, id="extra_out"),
        pytest.param(
            (),
            (symmetry.AxialDirRep,),
            symmetry.TrivialRep,
            id="extra_out_trivial_space",
        ),
    ],
)
def test_SymDecompLinear(extra_in_reps, extra_out_reps, out_space_rep):
    # prepare an example op
    inpf = SymDecompDims(17, 11, 5)
    outf = SymDecompDims(19, 13, 7, rep=RepSpec(out_space_rep, 10))

    rngs = nnx.Rngs(42)
    lin = SymDecompLinear(
        inpf,
        outf,
        extra_in_reps=extra_in_reps,
        extra_out_reps=extra_out_reps,
        rngs=rngs,
        kernel_init=quant,
        bias_init=quant,
    )

    # Build batch shape: semantic batch (3,) + extra_in_reps dimensions
    batch_shape = (3,) + tuple(len(rep) for rep in extra_in_reps)

    # verify symmetry
    inp = SplitSymDecomp.empty(inpf, batch=batch_shape)
    inp = attrs.evolve(
        inp,
        **{
            k: dict(invariant=1, space=8, flavour=10)[k]
            * jax.random.randint(rngs.params(), v.shape, -3, 4)
            / 2
            for k, v in inp.representations.items()
        },
    )
    out = lin(inp)

    for op in D4:
        # Compute transformation indices for extra reps and output space
        eitfo = tuple(
            symmetry.transform_coeff_in_basis(op, rep) for rep in extra_in_reps
        )
        eotfo = tuple(
            symmetry.transform_coeff_in_basis(op, rep) for rep in extra_out_reps
        )
        itfo = symmetry.transform_coeff_in_basis(op, symmetry.FullRep)
        otfo = symmetry.transform_coeff_in_basis(op, out_space_rep)

        # Transform input by applying extra_in transformations, then space transformation
        tinp = inp
        # Apply extra_in transformations sequentially
        for i, tfo in enumerate(eitfo):
            axis = 1 + i  # After semantic batch dimension
            tinp = attrs.evolve(
                tinp,
                invariant=jnp.take(tinp.invariant, tfo, axis=axis),
                flavour=jnp.take(tinp.flavour, tfo, axis=axis),
                space=jnp.take(tinp.space, tfo, axis=axis),
            )

        # Transform space representation
        tinp = attrs.evolve(
            tinp,
            space=jnp.take(tinp.space, itfo, axis=-2),
        )

        assert inpf.validate(tinp), inpf.validation_problems(tinp)

        # Transform expected output
        expected = out
        # Apply extra_out transformations sequentially
        for i, tfo in enumerate(eotfo):
            axis = 1 + i
            expected = attrs.evolve(
                expected,
                invariant=jnp.take(expected.invariant, tfo, axis=axis),
                flavour=jnp.take(expected.flavour, tfo, axis=axis),
                space=jnp.take(expected.space, tfo, axis=axis),
            )

        # Transform output space representation
        expected = attrs.evolve(
            expected,
            space=jnp.take(expected.space, otfo, axis=-2),
        )

        assert outf.validate(expected), outf.validation_problems(expected)

        actual = lin(tinp)
        for k, v in expected.representations.items():
            a = getattr(actual, k)
            assert np.allclose(a, v), f"{op}: {k}"

        verify_swap(op, tinp, expected, rngs, lin)


@pytest.mark.parametrize(
    "extra_in_reps,extra_out_reps,out_space_rep",
    [
        pytest.param((), (), symmetry.FullRep, id="no_extra"),
        pytest.param((symmetry.AxialDirRep,), (), symmetry.FullRep, id="extra_in"),
        pytest.param((), (symmetry.AxialDirRep,), symmetry.FullRep, id="extra_out"),
        pytest.param(
            (),
            (symmetry.AxialDirRep,),
            symmetry.TrivialRep,
            id="extra_out_trivial_space",
        ),
    ],
)
def test_SymDecompLinear_flat_vs_split(extra_in_reps, extra_out_reps, out_space_rep):
    """Test that flat and split modes produce identical results."""
    # prepare an example op
    inpf = SymDecompDims(17, 11, 5)
    outf = SymDecompDims(19, 13, 7, rep=RepSpec(out_space_rep, 10))

    rngs = nnx.Rngs(42)
    lin = SymDecompLinear(
        inpf,
        outf,
        extra_in_reps=extra_in_reps,
        extra_out_reps=extra_out_reps,
        rngs=rngs,
        kernel_init=quant,
        bias_init=quant,
    )

    # Build batch shape: semantic batch (3,) + extra_in_reps dimensions
    batch_shape = (3,) + tuple(len(rep) for rep in extra_in_reps)

    # Create test input
    inp = SplitSymDecomp.empty(inpf, batch=batch_shape)
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
    out_split = lin(inp, mode="split")

    # Apply with flat mode
    out_flat = lin(inp, mode="flat")

    # Verify both modes produce identical results (with tolerance for floating point rounding)
    for k, v_flat in out_flat.representations.items():
        v_split = getattr(out_split, k)
        assert np.allclose(
            v_split, v_flat, rtol=1e-5, atol=1e-5
        ), f"{k}: split and flat modes differ"
