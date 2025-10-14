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
