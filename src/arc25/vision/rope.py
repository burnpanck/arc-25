from types import SimpleNamespace

import attrs
import jax.nn
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


@attrs.frozen
class QKV:
    query: jt.Float[jt.Array, "... T P N H 2"]
    key: jt.Float[jt.Array, "... S P K H 2"]
    value: jt.Float[jt.Array, "... S P K D"]
    mask: jt.Bool[jt.Array, "... S"] | None = None

    @property
    def shape(self):
        T, P, N, H, two = self.query.shape[-5:]
        S = self.key.shape[-5]
        K = self.key.shape[-3]
        D = self.value.shape[-1]
        return SimpleNamespace(
            T=T,
            P=P,
            N=N,
            H=H,
            S=S,
            K=K,
            D=D,
            batch=self.query.shape[:-5],
        )

    def validation_problems(self):
        T, P, N, H, two = self.query.shape[-5:]
        S = self.key.shape[-5]
        K = self.key.shape[-3]
        D = self.value.shape[-1]
        if two != 2:
            return "query 2"
        if self.key.shape[-5:] != (S, P, K, H, 2):
            return "key"
        if self.value.shape[-4:] != (S, P, K, D):
            return "value"
        if self.mask.shape[-1] != S:
            return "mask"
        try:
            np.broadcast_shapes(
                self.query.shape[:-5],
                self.key.shape[:-5],
                self.value.shape[:-4],
                self.mask.shape[:-1],
            )
        except ValueError:
            return "batch"

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
    return f"({",".join(ret)})"


def attention_RoPE_with_global(
    globl: QKV,
    axial: QKV,
    pQ: jt.Float[jt.Array, "... T K H"],
    pK: jt.Float[jt.Array, "... S K H"] | None = None,
    *,
    # this one is usually static; values are 0: normal, 1: reverse
    polarisation: jt.Int[jt.Array, " P"],
):
    print(f"{globl.shape=} {axial.shape=} {pQ.shape=}")
    assert (
        globl.is_valid()
    ), f"{globl.validation_problems()}: q={globl.query.shape} k={globl.key.shape} v={globl.value.shape}"
    assert axial.is_valid(), axial.validation_problems()

    # global and axial need to be mostly consistent
    sa = axial.shape
    sg = globl.shape
    for k, v in vars(sa).items():
        if k in {"S", "T"}:
            continue
        assert getattr(sg, k) == v, f"{k}: {getattr(globl, k)} <> {v}"
    assert pQ.shape[-3:] == (sa.T, sa.K, sa.H), f"{pQ.shape=} {sa}"
    mpK = pK if pK is not None else pQ
    assert mpK.shape[-3:] == (sa.S, sa.K, sa.H), f"{pK.shape=} {sa}"
    try:
        np.broadcast_shapes(sa.batch, pQ.shape[:-3], mpK.shape[:-3])
    except ValueError:
        raise AssertionError(
            f"pQ and pK need to be broadcastable to the batch shape: {sa.batch=} {pQ.shape=} {mpK.shape=}"
        )

    # calculate rotation matrices; these have shape [... S/T P K H 2 2]: (length, polarisation, head, feature, u, v)
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
        print(f"{p.shape=} {rd.shape=} {r.shape=}")
        # r now will have the final target shape
        r = r[..., polarisation, :, :, :, :]
        print(f"{polarisation.shape=} {r.shape=}")
        phi.append(r)
    rQ, rK = phi

    Sa, P, K, H = axial.key.shape[-5:-1]
    Sg, _, D = globl.value.shape[-3:]
    Ta, _, Na = axial.query.shape[-5:-2]
    Tg, _, Ng = globl.query.shape[-5:-2]
    assert not Na % K
    assert not Ng % K
    Ma = Na // K
    Mg = Ng // K

    aQ = jnp.einsum(
        "...tpmkhu, ...tpkhuv -> ...tpmkhv",
        axial.query.reshape(*axial.query.shape[:-5], Ta, P, Ma, K, H, 2),
        rQ,
    )
    aK = jnp.einsum("...spkhu, ...spkhuv -> ...spkhv", axial.key, rK)
    aV = axial.value

    gQ = globl.query.reshape(*globl.query.shape[:-5], Tg, P, Mg, K, H, 2)
    gK = globl.key
    gV = globl.value

    print("aQ:", show_dims("tpmkhv", aQ))
    print("gQ:", show_dims("tpmkhv", gQ))
    print("aK:", show_dims("spkhv", aK))
    print("gK:", show_dims("spkhv", gK))

    log_aa = jnp.einsum("...tpmkhv,...spkhv -> ...pmkts", aQ, aK)
    log_gg = jnp.einsum("...tpmkhv,...spkhv -> ...pmkts", gQ, gK)
    log_ga = jnp.einsum("...tpmkhv,...spkhv -> ...pmkts", gQ, aK)
    log_ag = jnp.einsum("...tpmkhv,...spkhv -> ...pmkts", aQ, gK)

    scale = 1 / np.sqrt(H)
    value = jnp.concatenate([gV, aV], axis=-4)
    msh = np.broadcast_shapes(
        *[arg.mask.shape[:-1] for arg in [globl, axial] if arg.mask is not None]
    )
    # print(f"{msh=}")
    # print("globl.mask:",show_dims("s",globl.mask))
    # print("axial.mask:",show_dims("s",axial.mask))
    msk = (
        jnp.concatenate(
            [
                jnp.ones(msh + arg.value.shape[-4:-3]) if arg.mask is None else arg.mask
                for arg in [globl, axial]
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
        # print(f"{logits.shape=} ({show_dims("pmkts",logits)}) {msk.shape=}")
        prob = jax.nn.softmax(logits * scale, axis=-1, where=msk)
        # print("P:",show_dims("pmkts",prob))
        # print("V:",show_dims("spkd",value))
        result = jnp.einsum("...pmkts,...spkd -> ...tpmkd", prob, value)
        # print("result:",show_dims("tpmkd",result))
        result = result.reshape(*result.shape[:-3], -1, D)
        # print(f"rrs: {show_dims("tpnd",result)}, {Tg+Ta=} {P=} {N=} {D=}")
        assert result.shape[-4:] == (Tg + Ta, P, N, D)
        globl = result[..., :Tg, :, :, :]
        axial = result[..., Tg:, :, :, :]
    else:
        res = []
        for logits in [[log_gg, log_ga], [log_ag, log_aa]]:
            logits = jnp.concatenate(logits, axis=-1)
            prob = jax.nn.softmax(logits * scale, axis=-1, where=msk)
            result = jnp.einsum("...pmkts,...spkd -> ...tpmkd", prob, value)
            result = result.reshape(*result.shape[:-3], -1, D)
            res.append(result)
        globl, axial = res
    return globl, axial
