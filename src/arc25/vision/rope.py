from types import SimpleNamespace

import attrs
import jax.nn
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

from ..lib.attrs import AttrsModel


class QKV(AttrsModel):
    query: jt.Float[jt.Array, "... T F P N H 2"]
    key: jt.Float[jt.Array, "... S F P K H 2"]
    value: jt.Float[jt.Array, "... S F P K D"]
    mask: jt.Bool[jt.Array, "... S F"] | None = None

    @property
    def shape(self):
        T, F, P, N, H, two = self.query.shape[-6:]
        S = self.key.shape[-6]
        K = self.key.shape[-3]
        D = self.value.shape[-1]
        return SimpleNamespace(
            T=T,
            F=F,
            P=P,
            N=N,
            H=H,
            S=S,
            K=K,
            D=D,
            batch=self.query.shape[:-6],
        )

    def validation_problems(self):
        T, F, P, N, H, two = self.query.shape[-6:]
        S = self.key.shape[-6]
        K = self.key.shape[-3]
        D = self.value.shape[-1]
        if two != 2:
            return "query 2"
        if self.key.shape[-6:] != (S, F, P, K, H, 2):
            return "key"
        if self.value.shape[-5:] != (S, F, P, K, D):
            return "value"
        if self.mask is not None and self.mask.shape[-2:] != (S, F):
            return f"mask {self.mask.shape=}, expected {(S, F)} last"
        try:
            non_batch_dims = dict(
                query=6,
                key=6,
                value=5,
                mask=2,
            )
            np.broadcast_shapes(
                *[
                    a.shape[:-n]
                    for k, n in non_batch_dims.items()
                    if (a := getattr(self, k)) is not None
                ]
            )
        except ValueError:
            return "batch: " + " ".join(
                f"{k}:{getattr(self, k).shape[:-n]}" for k, n in non_batch_dims.items()
            )

    def is_valid(self):
        return not self.validation_problems()


def show_dims(dimnames: str, obj) -> str:
    try:
        shape = obj.shape
    except AttributeError:
        shape = obj
    batch = shape[: -len(dimnames)]
    ret = [str(n) for n in batch] + [
        f"{k}={v}" for k, v in zip(dimnames, shape[-len(dimnames) :])
    ]
    ret = ",".join(ret)
    return f"({ret})"


def attention_RoPE_with_global(
    context: QKV,
    axial: QKV,
    pQ: jt.Float[jt.Array, "... T K H"],
    pK: jt.Float[jt.Array, "... S K H"] | None = None,
    *,
    # this one is usually static; values are 0: normal, 1: reverse
    polarisation: jt.Int[jt.Array, " P"],
):
    # print(f"{context.shape=} {axial.shape=} {pQ.shape=}")
    assert context.is_valid(), context.validation_problems()
    assert axial.is_valid(), axial.validation_problems()

    # global and axial need to be mostly consistent
    sa = axial.shape
    sg = context.shape
    for k, v in vars(sa).items():
        if k in {"S", "T"}:
            continue
        assert getattr(sg, k) == v, f"{k}: {getattr(sg, k)} <> {v}"
    assert pQ.shape[-3:] == (sa.T, sa.K, sa.H), f"{pQ.shape=} {sa}"
    mpK = pK if pK is not None else pQ
    assert mpK.shape[-3:] == (sa.S, sa.K, sa.H), f"{pK.shape=} {sa}"
    try:
        np.broadcast_shapes(sa.batch, pQ.shape[:-3], mpK.shape[:-3])
    except ValueError:
        raise AssertionError(
            f"pQ and pK need to be broadcastable to the batch shape: {sa.batch=} {pQ.shape=} {mpK.shape=}"
        )

    # calculate rotation matrices; these have shape [... S/T F P K H 2 2]: (length, flavour, polarisation, head, feature, u, v)
    phi = []
    for p in [pQ, pK]:
        if p is None:
            phi.append(phi[-1])
            continue
        cs, sn = jnp.cos(p), jnp.sin(p)
        nsn = -sn
        # rd will have shape ... S/T K H 3
        rd = jnp.moveaxis(jnp.array([cs, sn, nsn]), 0, -1)
        # idx will have shape 2 2 2
        idx = np.r_[0, 2, 1, 0, 0, 1, 2, 0].reshape(2, 2, 2)
        # r will have shape ... S/T 2 K H 2 2
        r = jnp.moveaxis(rd[..., idx], -3, -5)
        # print(f"{p.shape=} {rd.shape=} {r.shape=}")
        # r now will have the final target shape
        r = r[..., None, polarisation, :, :, :, :]
        # print(f"{polarisation.shape=} {r.shape=}")
        phi.append(r)
    rQ, rK = phi

    Sa, F, P, K, H = axial.key.shape[-6:-1]
    Sg, _, _, D = context.value.shape[-4:]
    Ta, _, _, Na = axial.query.shape[-6:-2]
    Tg, _, _, Ng = context.query.shape[-6:-2]
    assert not Na % K
    assert not Ng % K
    Ma = Na // K
    Mg = Ng // K

    aQ = jnp.einsum(
        "...tfpmkhu, ...tfpkhuv -> ...tfpmkhv",
        axial.query.reshape(*axial.query.shape[:-6], Ta, F, P, Ma, K, H, 2),
        rQ,
    )

    aK = jnp.einsum("...sfpkhu, ...sfpkhuv -> ...sfpkhv", axial.key, rK)
    aV = axial.value

    gQ = context.query.reshape(*context.query.shape[:-6], Tg, F, P, Mg, K, H, 2)
    gK = context.key
    gV = context.value

    if False:
        print("aQ:", show_dims("tfpmkhv", aQ))
        print("gQ:", show_dims("tfpmkhv", gQ))
        print("aK:", show_dims("sfpkhv", aK))
        print("gK:", show_dims("sfpkhv", gK))

    log_aa = jnp.einsum("...tfpmkhv,...sfpkhv -> ...fpmkts", aQ, aK)
    log_gg = jnp.einsum("...tfpmkhv,...sfpkhv -> ...fpmkts", gQ, gK)
    log_ga = jnp.einsum("...tfpmkhv,...sfpkhv -> ...fpmkts", gQ, aK)
    log_ag = jnp.einsum("...tfpmkhv,...sfpkhv -> ...fpmkts", aQ, gK)

    scale = 1 / np.sqrt(H)
    value = jnp.concatenate([gV, aV], axis=-5)
    msh = np.broadcast_shapes(
        *[arg.mask.shape[:-2] for arg in [context, axial] if arg.mask is not None]
    )
    if False:
        print(f"{msh=}")
        if context.mask is not None:
            print("context.mask:", show_dims("sf", context.mask))
        else:
            print(
                "context.mask:",
                show_dims("sf", jnp.ones(msh + context.value.shape[-5:-3])),
            )
        print("axial.mask:", show_dims("sf", axial.mask))
    msk = (
        jnp.concatenate(
            [
                jnp.swapaxes(
                    (
                        jnp.ones(msh + arg.value.shape[-5:-3])
                        if arg.mask is None
                        else arg.mask
                    ),
                    -1,
                    -2,
                )
                for arg in [context, axial]
            ],
            axis=-1,
        )[..., None, None, None, None, :]
        if msh
        else None
    )
    # print(f"{Ma=} {Mg=} {Na=} {Ng=}")
    if Ma == Mg:
        assert K * Ma == Na == Ng == K * Mg
        N = Na
        logits = jnp.block([[log_gg, log_ga], [log_ag, log_aa]])
        # print(f"{logits.shape=} ({show_dims("fpmkts", logits)}) {msk.shape=} ({show_dims("fpmkts", msk)})")
        prob = jax.nn.softmax(logits * scale, axis=-1, where=msk)
        # print("P:",show_dims("pmkts",prob))
        # print("V:",show_dims("spkd",value))
        result = jnp.einsum("...fpmkts,...sfpkd -> ...tfpmkd", prob, value)
        # print("result:",show_dims("tpmkd",result))
        result = result.reshape(*result.shape[:-3], -1, D)
        # print(f"rrs: {show_dims("tpnd",result)}, {Tg+Ta=} {P=} {N=} {D=}")
        assert result.shape[-5:] == (Tg + Ta, F, P, N, D)
        context = result[..., :Tg, :, :, :, :]
        axial = result[..., Tg:, :, :, :, :]
    else:
        res = []
        for logits in [[log_gg, log_ga], [log_ag, log_aa]]:
            logits = jnp.concatenate(logits, axis=-1)
            prob = jax.nn.softmax(logits * scale, axis=-1, where=msk)
            result = jnp.einsum("...fpmkts,...sfpkd -> ...tfpmkd", prob, value)
            result = result.reshape(*result.shape[:-3], -1, D)
            res.append(result)
        context, axial = res
    return context, axial
