import contextlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import attrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .dsl.types import Canvas, Color, Image, MaskedImage, Paintable

# ARC color map - colors for values 0-9
_cmap = mpl.colors.ListedColormap([c.value for c in Color])
_norm = mpl.colors.Normalize(vmin=0, vmax=9)


def show_image(img: Paintable, *, ax=None):
    if ax is None:
        ax = plt.gca()
    if isinstance(img, Canvas):
        # TODO: mark physical axes in plot
        img = img.image
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
    )
    ax.yaxis.set_inverted(True)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect("equal", "box")
    return ax


def show_test_case(test_case):
    pairs = test_case.train + test_case.test
    N = len(pairs)
    fig, axes = plt.subplots(2, N, figsize=(12, 2 * 12 / N))
    for i, (p, axe) in enumerate(zip(pairs, axes.T)):
        c = "green" if i < len(test_case.train) else "red"
        for k, ax in zip(["input", "output"], axe):
            v = getattr(p, k, None)
            if v is None:
                continue
            show_image(v, ax=ax)
            for spine in ax.spines.values():
                spine.set_edgecolor(c)


@attrs.frozen
class IOPair:
    input: Canvas
    output: Canvas


@attrs.frozen
class Challenge:
    id: str
    train: tuple[IOPair, ...]
    test: tuple[IOPair | Canvas, ...]


def parse_inputs(v, had_list=False, id=None):
    match v:
        case dict():
            v = {kk: parse_inputs(vv, had_list) for kk, vv in v.items()}
            if had_list:
                return IOPair(**v)
            else:
                assert id is not None
                return Challenge(id=id, **v)
        case list():
            if not had_list:
                return tuple(parse_inputs(vv, True) for vv in v)
            else:
                return Canvas(Image(np.array(v, dtype="i1")))
        case _:
            raise TypeError(f"Unsupported type {type(v).__name__}")


@contextlib.contextmanager
def load_file(root, relative, mode="r"):
    match root.suffix:
        case ".zip":
            with zipfile.ZipFile(root, "r") as zfh:
                fh = zfh.open(relative)
                yield fh
        case "":
            with open(root / relative, mode) as fh:
                yield fh
        case _:
            raise KeyError(f"Unknown suffix {root.suffix!r}")


def load_dataset(
    root: Path, challenges: Path, solutions: Path | None = None
) -> dict[str, Challenge]:
    with load_file(root, challenges, "rt") as fh:
        challenges = {k: parse_inputs(v, id=k) for k, v in json.load(fh).items()}
    if solutions is not None:
        with load_file(root, solutions, "rt") as fh:
            solutions = {k: parse_inputs(v) for k, v in json.load(fh).items()}
        dataset = {}
        for k, v in challenges.items():
            dataset[k] = Challenge(
                train=v.train,
                test=tuple(
                    IOPair(
                        input=i.input,
                        output=o,
                    )
                    for i, o in zip(v.test, solutions[k])
                ),
            )
    else:
        dataset = challenges
    return dataset
