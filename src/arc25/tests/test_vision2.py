import attrs
import jax
import numpy as np
import pytest
from flax import nnx

from .. import symmetry
from ..lib.attrs import AttrsModel
from ..symmetry import D4
from ..vision2 import attention
from ..vision2.linear import SymDecompLinear
from ..vision2.symrep import RepSpec, SplitSymDecomp, SymDecompDims


def quant(key, shape, dtype):
    seed = np.array(jax.random.key_data(key)).astype("u4")
    seed = (seed[0] ^ seed[1]) & 0x7FFF_FFFF
    rng = np.random.RandomState(seed)
    return rng.randint(-3, 4, size=shape).astype(dtype) / 2


def verify_swap(name, inp, expected, rngs, fun, *, n_swaps=10):
    for swapiter in range(n_swaps):
        swp = np.r_[:10]
        i, j = jax.random.randint(rngs(), 2, 0, 10)
        swp[[i, j]] = swp[[j, i]]
        inp = attrs.evolve(
            inp,
            flavour=inp.flavour[..., swp, :],
        )
        expected = attrs.evolve(
            expected,
            flavour=expected.flavour[..., swp, :],
        )
        actual = fun(inp)
        for k, v in expected.representations.items():
            a = getattr(actual, k)
            assert np.allclose(a, v), f"{name}/{swapiter} ({swp}): {k}"


def test_SymDecompLinear1():
    # prepare an example op
    inpf = SymDecompDims(17, 11, 5)
    outf = SymDecompDims(19, 13, 7)

    rngs = nnx.Rngs(42)
    lin = SymDecompLinear(
        inpf,
        outf,
        extra_in_reps=(symmetry.AxialDirRep,),
        rngs=rngs,
        kernel_init=quant,
        bias_init=quant,
    )

    # verify symmetry
    inp = SplitSymDecomp.empty(inpf, batch=(3, 4))
    for k, v in inp.representations.items():
        s = dict(invariant=1, space=8, flavour=10)[k]
        v[...] = s * jax.random.randint(rngs.params(), v.shape, -3, 4) / 2
    out = lin(inp)
    for op in D4:
        atfo, ftfo = [
            symmetry.transform_coeff_in_basis(op, rep)
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

        verify_swap(op, tinp, expected, rngs, lin)


def test_SymDecompLinear2():
    # prepare an example op
    inpf = SymDecompDims(17, 11, 5)
    outf = SymDecompDims(19, 13, 7)

    rngs = nnx.Rngs(42)
    lin = SymDecompLinear(
        inpf,
        outf,
        extra_out_reps=(symmetry.AxialDirRep,),
        rngs=rngs,
        kernel_init=quant,
        bias_init=quant,
    )

    # verify symmetry
    inp = SplitSymDecomp.empty(inpf, batch=(3,))
    for k, v in inp.representations.items():
        s = dict(invariant=1, space=8, flavour=10)[k]
        v[...] = s * jax.random.randint(rngs.params(), v.shape, -3, 4) / 2
    out = lin(inp)
    for op in D4:
        atfo, ftfo = [
            symmetry.transform_coeff_in_basis(op, rep)
            for rep in [symmetry.AxialDirRep, symmetry.FullRep]
        ]
        tinp = attrs.evolve(
            inp,
            space=inp.space[:, ftfo, :],
        )
        assert inpf.validate(tinp), inpf.validation_problems(tinp)
        expected = attrs.evolve(
            out,
            invariant=out.invariant[:, atfo, :],
            flavour=out.flavour[:, atfo, :, :],
            space=out.space[:, atfo[:, None], ftfo, :],
        )
        assert outf.validate(expected), outf.validation_problems(expected)
        actual = lin(tinp)
        for k, v in expected.representations.items():
            a = getattr(actual, k)
            assert np.allclose(a, v), f"{op}: {k}"
        verify_swap(op, tinp, expected, rngs, lin)


def test_SymDecompLinear3():
    # prepare an example op
    inpf = SymDecompDims(17, 11, 5)
    outf = SymDecompDims(19, 13, 7, rep=RepSpec(symmetry.TrivialRep, 10))

    rngs = nnx.Rngs(42)
    lin = SymDecompLinear(
        inpf,
        outf,
        extra_out_reps=(symmetry.AxialDirRep,),
        rngs=rngs,
        kernel_init=quant,
        bias_init=quant,
    )

    # verify symmetry
    inp = SplitSymDecomp.empty(inpf, batch=(3,))
    for k, v in inp.representations.items():
        s = dict(invariant=1, space=8, flavour=10)[k]
        v[...] = s * jax.random.randint(rngs.params(), v.shape, -3, 4) / 2
    out = lin(inp)
    for op in D4:
        atfo, ftfo = [
            symmetry.transform_coeff_in_basis(op, rep)
            for rep in [symmetry.AxialDirRep, symmetry.FullRep]
        ]
        tinp = attrs.evolve(
            inp,
            space=inp.space[:, ftfo, :],
        )
        assert inpf.validate(tinp), inpf.validation_problems(tinp)
        expected = attrs.evolve(
            out,
            invariant=out.invariant[:, atfo, :],
            flavour=out.flavour[:, atfo, :, :],
            space=out.space[:, atfo, :, :],
        )
        assert outf.validate(expected), outf.validation_problems(expected)
        actual = lin(tinp)
        for k, v in expected.representations.items():
            a = getattr(actual, k)
            assert np.allclose(a, v), f"{op}: {k}"
        verify_swap(op, tinp, expected, rngs, lin)


def test_RoPE():
    with jax.experimental.enable_x64():
        # prepare an example attention module with prime dimensions
        inpf = SymDecompDims(17, 11, 5)
        outf = SymDecompDims(19, 13, 7)

        atn = attention.AxialAttention(
            num_heads=6,
            num_groups=2,
            in_features=inpf,
            out_features=outf,
            qk_head_width=SymDecompDims(
                13 * 2, 23 * 2, 3 * 2, rep=RepSpec(symmetry.TrivialRep, 10)
            ),
            v_head_width=SymDecompDims(5, 4, 2),
            use_chirality_rep=True,
            kernel_init=quant,
            bias_init=quant,
            dtype=np.float64,
            activation=jax.nn.sigmoid,
            use_bias=False,
            rngs=nnx.Rngs(42),
        )

        # prepare field inputs with prime spatial dimensions
        Y, X = 3, 4  # prime dimensions for spatial field
        inp = SplitSymDecomp.empty(inpf, batch=(2, Y, X))
        n_batch_dims = 1
        for k, v in inp.representations.items():
            s = dict(invariant=1, space=8, flavour=10)[k]
            v[...] = s * jax.random.randint(atn.rngs.params(), v.shape, -3, 4) / 2

        # prepare coordinate grid
        coords = attention.CoordinateGrid.from_shape(*inp.batch_shape[-2:])

        out, atnw = atn(inp, coords, return_attention_weights=True)

        # verify independence of coordinate offsets
        ncoords = attrs.evolve(
            coords,
            xpos=coords.xpos + [X, 1.0],
            ypos=coords.ypos - [Y, 1.5],
        )
        print(coords.xpos.T)
        print(ncoords.xpos.T)

        actual, tatnw = atn(inp, ncoords, return_attention_weights=True)

        # verify the attention-weights match
        for axis, v in enumerate(atnw):
            a = tatnw[axis]
            max_diff = np.abs(a - v).max()
            rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
            bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
            n_failed = int(bad.sum())
            assert np.allclose(
                a, v, rtol=1e-4, atol=1e-3
            ), f"@{axis} attention weights: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}"
            print(f"@{axis} attention weights: {max_diff=:.3e}, {rel_diff=:.5f}")

        # verify all representations match
        for k, v in out.representations.items():
            a = getattr(actual, k)
            max_diff = np.abs(a - v).max()
            rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
            bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
            n_failed = int(bad.sum())
            bad_idx = np.array(np.nonzero(bad)).T[:4]
            bad_info = []
            for idx in bad_idx:
                idx = tuple(int(v) for v in idx)
                av = a[idx]
                ev = v[idx]
                b, idx = idx[:n_batch_dims], idx[n_batch_dims:]
                y, x = idx[:2]
                c = idx[-1]
                f = idx[2:-1]
                match k:
                    case "invariant":
                        assert not f
                        f = ""
                    case "space":
                        (f,) = f
                        f = list(outf.rep.space)[f].symbol
                    case "flavour":
                        (f,) = f
                        f = str(f)
                bad_info.append(
                    f"{b=} {y=} {x=} {f=} {c=}: expected {ev}, actual {av},"
                    f" error {abs(ev-av)} ({abs(av-ev)/(abs(ev)+1e-8)} relative)"
                )
            bad_info = "\n".join(bad_info)
            assert np.allclose(
                a, v, rtol=1e-4, atol=1e-3
            ), f"@{k}: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}, bad:\n{bad_info}"
            print(f"@{k}: {max_diff=:.3e}, {rel_diff=:.5f}")


@pytest.mark.parametrize("use_chirality", [False, True])
def test_AxialAttention(use_chirality):
    with jax.experimental.enable_x64():
        # prepare an example attention module with prime dimensions
        inpf = SymDecompDims(17, 11, 5)
        outf = SymDecompDims(19, 13, 7)

        atn = attention.AxialAttention(
            num_heads=6,
            num_groups=2,
            in_features=inpf,
            out_features=outf,
            qk_head_width=SymDecompDims(
                13 * 2,
                23 * 2,
                3 * 2,
                rep=RepSpec(
                    symmetry.TrivialRep if use_chirality else symmetry.ChiralityRep, 10
                ),
            ),
            v_head_width=SymDecompDims(5, 4, 2),
            use_chirality_rep=use_chirality,
            kernel_init=quant,
            bias_init=quant,
            dtype=np.float64,
            activation=jax.nn.sigmoid,
            # use_bias=False,
            rngs=nnx.Rngs(42),
        )

        # prepare field inputs with prime spatial dimensions
        Y, X = 3, 4  # prime dimensions for spatial field
        inp = SplitSymDecomp.empty(inpf, batch=(2, Y, X))
        n_batch_dims = 1
        for k, v in inp.representations.items():
            s = dict(invariant=1, space=8, flavour=10)[k]
            v[...] = s * jax.random.randint(atn.rngs.params(), v.shape, -3, 4) / 2

        # prepare coordinate grid
        coords = attention.CoordinateGrid.from_shape(*inp.batch_shape[-2:])

        out, atnw = atn(inp, coords, return_attention_weights=True)

        # verify symmetry preservation
        for op in D4:
            # transform representation index for FullRep in space
            ftfo = symmetry.transform_coeff_in_basis(op, symmetry.FullRep)

            def tfo_image(image):
                # make sure we have batch dimensions respected
                return symmetry.transform_image(
                    op, image, ydim=n_batch_dims, xdim=n_batch_dims + 1  # noqa: B023
                )

            def tfo_field(field, dim):
                ret = attrs.evolve(
                    field,
                    invariant=tfo_image(field.invariant),
                    flavour=tfo_image(field.flavour),
                    space=tfo_image(field.space)[..., ftfo, :],  # noqa: B023
                )
                assert dim.validate(ret), dim.validation_problems(ret)
                return ret

            # transform spatial field
            tinp = tfo_field(inp, inpf)

            # rebuild coordinate grid for transformed shape
            tcoords = attention.CoordinateGrid.from_shape(*tinp.batch_shape[-2:])

            # expected attention weights:
            eatnw = [None] * 2
            for axis, (aw, dims) in enumerate(zip(atnw, ["tsxdvmk", "ytsdvmk"])):
                # first, prepare the token representation
                d = np.array(
                    [
                        symmetry.AxialDirRep(v).apply(op).index
                        for v in ["↓↑", "→←"][axis]
                    ]
                )
                v = (
                    np.array([v.apply(op).index for v in symmetry.ChiralityRep])
                    if atn.use_chirality_rep
                    else np.s_[:]
                )
                naxis = d // 2
                assert np.all(naxis == naxis.flat[0])
                naxis = naxis.flat[0]
                assert (naxis == axis) == (not op.value & D4.t.value)
                d = d % 2
                # now, apply all the representations per dimension (excluding the final spatial transpose)
                info = []
                for i, dim in enumerate(dims):
                    match dim:
                        case "t" | "s":
                            tfo = bool(op.value & [D4.y, D4.x][axis].value)
                        case "x":
                            tfo = bool(op.value & D4.x.value)
                        case "y":
                            tfo = bool(op.value & D4.y.value)
                        case "d":
                            tfo = d
                        case "v":
                            tfo = v
                        case _:
                            continue
                    if isinstance(tfo, bool):
                        if not tfo:
                            continue
                        tfo = np.s_[::-1]
                        info.append(f"{dim}:⇅")
                    else:
                        info.append(f"{dim}:{tfo}")
                    i = -len(dims) + i
                    aw = np.moveaxis(np.moveaxis(aw, i, 0)[tfo], 0, i)
                # final spatial transpose
                if op.value & D4.t.value:
                    if axis:
                        aw = np.moveaxis(aw, -7, -5)
                    else:
                        aw = np.moveaxis(aw, -5, -7)
                    info.append("T")
                print(f"{op}: axis:{axis}->{naxis} {' '.join(info)}")
                eatnw[naxis] = aw

            # expected output: transform the original output
            expected = tfo_field(out, outf)

            # actual output: apply attention to transformed input with rebuilt coordinates
            actual, tatnw = atn(tinp, tcoords, return_attention_weights=True)

            # verify the attention-weights match
            for axis, v in enumerate(eatnw):
                a = tatnw[axis]
                max_diff = np.abs(a - v).max()
                rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
                bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
                n_failed = int(bad.sum())
                assert np.allclose(
                    a, v, rtol=1e-4, atol=1e-3
                ), f"{op} @{axis} attention weights: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}"

            # verify all representations match
            for k, v in expected.representations.items():
                a = getattr(actual, k)
                max_diff = np.abs(a - v).max()
                rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
                bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
                n_failed = int(bad.sum())
                bad_idx = np.array(np.nonzero(bad)).T[:4]
                bad_info = []
                for idx in bad_idx:
                    idx = tuple(int(v) for v in idx)
                    av = a[idx]
                    ev = v[idx]
                    b, idx = idx[:n_batch_dims], idx[n_batch_dims:]
                    y, x = idx[:2]
                    c = idx[-1]
                    f = idx[2:-1]
                    match k:
                        case "invariant":
                            assert not f
                            f = ""
                        case "space":
                            (f,) = f
                            f = list(outf.rep.space)[f].symbol
                        case "flavour":
                            (f,) = f
                            f = str(f)
                    bad_info.append(
                        f"{b=} {y=} {x=} {f=} {c=}: expected {ev}, actual {av},"
                        f" error {abs(ev-av)} ({abs(av-ev)/(abs(ev)+1e-8)} relative)"
                    )
                bad_info = "\n".join(bad_info)
                assert np.allclose(
                    a, v, rtol=1e-4, atol=1e-3
                ), f"{op} @{k}: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}, bad:\n{bad_info}"
                print(f"{op} @{k}: {max_diff=:.3e}, {rel_diff=:.5f}")

            verify_swap(
                op,
                tinp,
                expected,
                atn.rngs,
                lambda i: atn(i, tcoords),  # noqa: B023
            )
