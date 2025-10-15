import attrs
import jax
import jax.numpy as jnp
import numpy as np


def quant(key, shape, dtype):
    """Quantized random initializer for testing."""
    seed = np.array(jax.random.key_data(key)).astype("u4")
    seed = (seed[0] ^ seed[1]) & 0x7FFF_FFFF
    rng = np.random.RandomState(seed)
    return rng.randint(-3, 4, size=shape).astype(dtype) / 2


def verify_swap(name, inp, expected, rngs, fun, *, n_swaps=10):
    """Verify that flavour permutation symmetry is preserved."""
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
            assert np.allclose(
                a, v, rtol=1e-5, atol=1e-5
            ), f"{name}/{swapiter} ({swp}): {k}"
