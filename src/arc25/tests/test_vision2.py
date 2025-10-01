import attrs
import jax
import numpy as np
from flax import nnx

from .. import symmetry
from ..symmetry import D4
from ..vision2.linear import SymDecompLinear
from ..vision2.symrep import SplitSymDecomp, SymDecompDims


def test_SymDecompLinear():
    # prepare an example op
    inpf = SymDecompDims(17, 11, 5)
    outf = SymDecompDims(19, 13, 7)

    def quant(key, shape, dtype):
        return (jax.random.randint(key, shape, -3, 4) / 2).astype(dtype)

    lin = SymDecompLinear(
        inpf,
        outf,
        extra_in_reps=(symmetry.AxialDirRep,),
        rngs=nnx.Rngs(42),
        kernel_init=quant,
        bias_init=quant,
    )

    # verify symmetry
    inp = SplitSymDecomp.empty(inpf, batch=(3, 4))
    for k, v in inp.representations.items():
        s = dict(invariant=1, space=8, flavour=10)[k]
        v[...] = s * jax.random.randint(lin.rngs.params(), v.shape, -3, 4) / 2
    out = lin(inp)
    for op in D4:
        atfo, ftfo = [
            symmetry.transform_rep_idx(op, rep)
            for rep in [symmetry.AxialDirRep, symmetry.FullRep]
        ]
        tinp = attrs.evolve(
            inp,
            invariant=inp.invariant[:, atfo, :],
            flavour=inp.flavour[:, atfo, :, :],
            space=inp.space[:, atfo[:, None], ftfo, :],
        )
        assert inpf.validate(tinp), inpf.validation_problems(tinp)
        expected = attrs.evolve(
            out,
            space=out.space[:, ftfo, :],
        )
        assert outf.validate(expected), outf.validation_problems(expected)
        actual = lin(tinp)
        for k, v in expected.representations.items():
            a = getattr(actual, k)
            assert np.allclose(a, v), f"{op}: {k}"
        for swapiter in range(10):
            swp = np.r_[:10]
            i, j = jax.random.randint(lin.rngs(), 2, 0, 10)
            swp[[i, j]] = swp[[j, i]]
            tinp = attrs.evolve(
                tinp,
                flavour=tinp.flavour[:, :, swp, :],
            )
            expected = attrs.evolve(
                expected,
                flavour=expected.flavour[:, swp, :],
            )
            actual = lin(tinp)
            for k, v in expected.representations.items():
                a = getattr(actual, k)
                assert np.allclose(a, v), f"{op}/{swapiter} ({swp}): {k}"
