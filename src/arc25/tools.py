import contextlib
import json
import re
import typing
import zipfile
from pathlib import Path
from types import SimpleNamespace

import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import squarify

from .augment import all_colors_rex
from .dataset import IAETriple, IOPair, ReasonedSolution
from .dsl.types import AnyImage, Color, Image, MaskedImage

# ARC color map - colors for values 0-9
_cmap = mpl.colors.ListedColormap([c.value for c in Color])
_norm = mpl.colors.Normalize(vmin=0, vmax=9)


def show_image(img: AnyImage, *, ax=None, **kw):
    if ax is None:
        ax = plt.gca()
    match img:
        case Image():
            img = img._data
        case MaskedImage():
            img = np.where(img._mask, img._data, np.nan)
        case np.ndarray():
            pass
        case _:
            raise TypeError(f"Unsupported image type {type(img).__qualname__}")
    ax.pcolormesh(
        img,
        cmap=_cmap,
        norm=_norm,
        ec="gray",
        lw=0.1,
        **kw,
    )
    ax.yaxis.set_inverted(True)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect("equal", "box")
    return ax


def show_prediction(
    prediction: np.ndarray,
    *,
    ground_truth: Image | None = None,
    ax=None,
    pad=0.05,
    **kw,
):
    if ax is None:
        ax = plt.gca()
    match ground_truth:
        case np.ndarray() | None:
            pass
        case Image():
            ground_truth = ground_truth._data
        case MaskedImage():
            ground_truth = np.where(ground_truth._mask, ground_truth._data, np.nan)
        case _:
            raise TypeError(f"Unsupported image type {type(ground_truth).__qualname__}")

    if ground_truth is not None:
        raise NotImplementedError("ground_truth")

    p = np.array(prediction)
    p *= 1 / p.sum(axis=-1, keepdims=True)

    if False:
        # plot base mesh
        img = np.tile(np.nan, p.shape[:-1])
        # just for the mesh
        ax.pcolormesh(
            img,
            ec="gray",
            lw=0.4,
            **kw,
        )

    seq = np.argsort(p, axis=-1)[..., ::-1]
    sorted_p = np.take_along_axis(p, seq, axis=-1)
    rects = [[] for _ in range(10)]
    first = [[] for _ in range(10)]
    s = 1 - 2 * pad
    clist = tuple(c.value for c in Color)
    for idx in np.ndindex(p.shape[:2]):
        y, x = idx
        a = sorted_p[idx]
        assert np.all(a[1:] <= a[:-1]), str(np.round(a * 100).astype(int))
        cell = squarify.squarify(
            a * s**2,
            x + pad,
            y + pad,
            s,
            s,
        )
        for j, (i, r) in enumerate(zip(seq[idx], cell)):
            (rects[i] if j else first[i]).append(
                mpl.patches.Rectangle((r["x"], r["y"]), r["dx"], r["dy"])
            )
    for rss, a in zip([rects, first], [0.5, 1]):
        for rs, c in zip(rss, clist):
            coll = mpl.collections.PatchCollection(rs, fc=c, ec="none", lw=0.2, alpha=a)
            ax.add_collection(coll)

    h, w = p.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.yaxis.set_inverted(True)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect("equal", "box")
    return ax


def show_test_case(
    examples: typing.Iterable[IOPair] | typing.Iterable[IAETriple],
    *,
    n_train: int | None = None,
    fig=None,
    width=None,
    example_width=None,
    orientation: typing.Literal["h", "v"] = "v",
):
    assert width is None or example_width is None
    examples = list(examples)
    N = len(examples)
    M = {IOPair: 2, IAETriple: 3}[type(examples[0])]
    if fig is None:
        fig = plt.gcf()
    if example_width is not None:
        match orientation:
            case "h":
                width = N * example_width
            case "v":
                width = M * example_width
    if width is None:
        width = 12
    match orientation:
        case "h":
            fig.set_size_inches(width, M * width / N)
            axes = fig.subplots(M, N)
        case "v":
            fig.set_size_inches(width, N * width / M)
            axes = fig.subplots(N, M).T
        case _:
            raise KeyError(orientation)
    for i, (e, axe) in enumerate(zip(examples, axes.T)):
        c = "black" if n_train is None else "green" if i < n_train else "red"
        for k, ax in zip(
            {
                2: ["input", "output"],
                3: ["input", "actual", "expected"],
            }[M],
            axe,
        ):
            v = getattr(e, k, None)
            if v is None:
                continue
            show_image(v, ax=ax)
            for spine in ax.spines.values():
                spine.set_edgecolor(c)


def solution_to_markdown(resp: ReasonedSolution):
    descr = "\n".join(f"### {k.title()}\n{v}\n" for k, v in resp.descr.items())
    body = "".join(
        f"## {k}\n{v}\n"
        for k, v in {
            "Description": descr,
            "Rule": resp.rule_descr,
            "Plan": resp.impl_plan_descr,
        }.items()
        if v
    )

    def extract_rgb(c: str):
        c = Color[c.upper()]
        v = int(c.value[1:], 16)
        ret = ",".join(str((v >> (k * 8)) % 256) for k in [2, 1, 0])
        return ret

    body = all_colors_rex.sub(
        lambda m: f'<span style="background-color:rgba({extract_rgb(m.group(1))},0.25)">{m.group(1)}</span>',
        body,
    )

    ret = f"""
{body}
```python
{resp.rule_impl}
```
""".strip()

    return ret
