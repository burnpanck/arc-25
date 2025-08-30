import contextlib
import dataclasses
import inspect
import itertools
import linecache
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Iterable, Literal, TypeAlias

import anyio
import attrs
import msgpack
import numpy as np

from .dataset import Challenge, IAETriple, IOPair, Solution
from .dsl import api as dsl
from .dsl import types
from .dsl.types import Canvas, Image, MaskedImage
from .symmetry import SymOp


@attrs.frozen(kw_only=True)
class ExecutionInfo:
    error: str | None = None
    stdout: tuple[str, ...] = ()
    stderr: tuple[str, ...] = ()


@attrs.frozen(kw_only=True)
class ExampleEval:
    output: Canvas | None = None
    # *_match is None if we don't know the correct output
    full_match: bool | None = None
    cell_match: float | None = None
    exec_info: ExecutionInfo | None = None


@attrs.frozen
class ChallengeEval:
    challenge: Challenge
    solution: Solution
    exec_info: ExecutionInfo
    train_eval: tuple[ExampleEval, ...]
    test_eval: tuple[ExampleEval, ...]
    full_match: bool
    example_match: float
    cell_match: float

    def get_eval_triples(
        self, subset: Literal["train", "test", "all"] = "all"
    ) -> Iterable[IAETriple]:
        for k in dict(all=["train", "test"]).get(subset, [subset]):
            for io, e in zip(getattr(self.challenge, k), getattr(self, f"{k}_eval")):
                yield io.compare_output(e.output)

    def any_error(self):
        return self.exec_info.error or any(
            e.exec_info.error for e in self.train_eval + self.test_eval
        )

    def summary(self, with_ansi: bool = True) -> list[str]:
        ret = []

        def output(s, colour, strong):
            modifiers = list()
            if with_ansi:
                if strong:
                    modifiers.append(1)
                #                    modifiers.append(47)
                if colour:
                    modifiers.append(
                        dict(
                            red=31,
                            green=32,
                            yellow=33,
                            orange=33,
                            blue=34,
                            magenta=35,
                            cyan=36,
                            white=37,
                        )[colour]
                    )
            csi = "\x1b["
            pre = (
                "" if not modifiers else csi + ";".join(str(v) for v in modifiers) + "m"
            )
            post = "" if not modifiers else csi + "0m"
            ret.append(pre + s + post)

        def ifp(s, colour=None, strong: bool = False):
            if s:
                output(s, colour, strong)

        ei = self.exec_info
        if not ei.error:
            output(
                f"Correct? {eval.full_match}, example fraction {eval.example_match*100:.0f} %,"
                f" cell fraction {eval.cell_match*100:.0f} %",
                colour="green" if eval.full_match else "orange",
                strong=True,
            )
        else:
            output(
                f"Error: {ei.error}",
                colour="red",
                strong=True,
            )
        ifp(ei.stdout)
        ifp(ei.stderr, colour="orange")
        for k in ["train", "test"]:
            for i, e in enumerate(getattr(self, f"{k}_eval"), 1):
                ei = e.exec_info
                if not ei.error:
                    output(
                        f"{k.title()} {i}: Correct? {e.full_match}, cell fraction {e.cell_match*100:.0f} %",
                        colour="green" if e.full_match else "orange",
                        strong=True,
                    )
                else:
                    output(
                        f"{k.title()} {i}: Error: {ei.error}",
                        colour="red",
                        strong=True,
                    )
                ifp(ei.stdout)
                ifp(ei.stderr, colour="orange")

        return ret


def _mark(id: str, mark: str):
    msg = f"\n<!-- [{id}|{mark}] -->\n"
    for stream in [sys.stdout, sys.stderr]:
        stream.write(msg)
        stream.flush()


def _evaluate_solution(challenge: Challenge, solution: Solution) -> dict:
    id = f"{challenge.id}"
    _mark(id, "load")
    try:
        filename = "<rule-code>"
        src = solution.rule
        code = compile(src, filename, "exec")
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        glob = {k: getattr(dsl, k) for k in dsl.__all__}
        glob["IOPair"] = IOPair
        loc = dict()
        exec(code, globals=glob, locals=loc)
        solver = loc.get("solution")
        if solver is None:
            raise ValueError("Rule must contain a function called `solution`")
        sig = inspect.signature(solver, eval_str=False)
        match len(sig.parameters):
            case 1:
                _real_solver = solver
                solver = lambda input, examples: _real_solver(input)
            case 2:
                pass
            case _:
                raise ValueError(
                    "`solution` must either take one argument `(input: Canvas)`,"
                    " or two arguments `(input: Canvas, examples:list[IOPair])`"
                )

        result = dict(train=[], test=[])
        examples = tuple(challenge.train) + tuple(
            attrs.evolve(v, output=None) for v in challenge.test
        )
        for eset in ["train", "test"]:
            for idx, io in enumerate(getattr(challenge, eset)):
                _mark(id, f"{eset}|{idx}")
                try:
                    actual = solver(io.input, examples)
                    if not isinstance(actual, types.Canvas):
                        raise TypeError(
                            f"`solution` must return a `Canvas`, got `{type(actual).__name__}`"
                        )
                except Exception as ex:
                    sys.stderr.write(traceback.format_exc())
                    eval = dict(error=repr(ex))
                else:
                    eval = dict(output=actual)
                result[eset].append(eval)
        _mark(id, "unload")
        return dict(results=result)
    except Exception as ex:
        sys.stderr.write(traceback.format_exc())
        err = repr(ex)
        return dict(error=err)


def serialise(obj):
    match obj:
        case dict():
            return {k: serialise(v) for k, v in obj.items()}
        case tuple() | list():
            return type(obj)(serialise(v) for v in obj)
        case np.ndarray():
            return dict(
                __type__=type(obj).__qualname__,
                shape=obj.shape,
                dtype=str(obj.dtype),
                data=bytes(obj.ravel()),
            )
        case SymOp():
            return dict(__type__=type(obj).__qualname__, name=obj.name)
        case _ if attrs.has(type(obj)):
            dct = attrs.asdict(obj, recurse=False)
            dct["__type__"] = type(obj).__qualname__
            return serialise(dct)
        case _ if dataclasses.is_dataclass(obj):
            dct = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
            dct["__type__"] = type(obj).__qualname__
            return serialise(dct)
        case _:
            return obj


_known_types = {
    t.__qualname__: t
    for t in [
        ChallengeEval,
        Challenge,
        Solution,
        Canvas,
        Image,
        MaskedImage,
        IOPair,
    ]
}


def deserialise(data):
    match data:
        case dict():
            typ = data.pop("__type__", None)
            if typ is None:
                return {k: deserialise(v) for k, v in data.items()}
        case tuple() | list():
            return type(data)(deserialise(v) for v in data)
        case _:
            return data
    match typ:
        case np.ndarray.__qualname__:
            return np.frombuffer(data["data"], dtype=data["dtype"]).reshape(
                *data["shape"]
            )
        case SymOp.__qualname__:
            return SymOp[data["name"]]
    cls = _known_types.get(typ)
    if cls is None:
        raise TypeError(f"Unsupported type {typ}")
    return cls(**deserialise(data))


async def evaluate_solution(challenge: Challenge, solution: Solution) -> ChallengeEval:
    async with contextlib.AsyncExitStack() as stack:
        tdir = anyio.Path(stack.enter_context(tempfile.TemporaryDirectory()))
        tdir = await tdir.resolve()
        input_file = tdir / "input.bson"
        output_file = tdir / "output.bson"
        await input_file.write_bytes(
            msgpack.packb(serialise(dict(challenge=challenge, solution=solution)))
        )
        with anyio.fail_after(1):
            result = await anyio.run_process(
                [
                    sys.executable,
                    "-m",
                    __name__,
                    "-i",
                    str(input_file),
                    "-o",
                    str(output_file),
                ]
            )
        out = await output_file.read_bytes()
    rex = re.compile(
        r"<!-- \["
        + re.escape(challenge.id)
        + r"\|(load|unload|train\|\d+|test\|\d+)\] -->$"
    )
    exec_info = dict(stdout=[], stderr=[])
    split_info = {}
    for k, v in dict(
        stdout=result.stdout.decode(),
        stderr=result.stderr.decode(),
    ).items():
        cur = "preload"
        split = {cur: []}
        for line in v.split("\n"):
            m = rex.match(line)
            if m:
                # remove last line if it is empty
                for v in [split[cur], exec_info[k]]:
                    if v and not v[-1]:
                        del v[-1]
                cur = m.group(1)
                split[cur] = []
                continue
            split.setdefault(cur, []).append(line)
            exec_info[k].append(line)
        split_info[k] = split
    out = msgpack.unpackb(out)
    output = deserialise(out)
    meta = dict(
        challenge=challenge,
        solution=solution,
    )
    err = output.pop("error", None)
    if err:
        assert not output
        exec_info = ExecutionInfo(
            error=err, **{k: "\n".join(v) for k, v in exec_info.items()}
        )
        return ChallengeEval(
            **meta,
            exec_info=exec_info,
            train_eval=challenge.get_empty_eval_triples("train"),
            test_eval=challenge.get_empty_eval_triples("test"),
            full_match=False,
            example_match=0,
            cell_match=0,
        )

    results = output.pop("results")
    assert not output

    exec_info = ExecutionInfo(
        **{
            k: "\n".join(
                itertools.chain(*[sv[sk] for sk in ["preload", "load", "unload"]])
            )
            for k, sv in split_info.items()
        }
    )

    splits = dict(train=[], test=[])
    for split, outputs in results.items():
        for idx, (io, res) in enumerate(zip(getattr(challenge, split), outputs)):
            err = res.pop("error", None)
            exec_info = ExecutionInfo(
                **{
                    k: "\n".join(v.get(f"{split}|{idx}", []))
                    for k, v in split_info.items()
                },
                error=err,
            )
            if err is not None:
                assert not output
                splits[split].append(
                    ExampleEval(
                        output=None,
                        full_match=False,
                        cell_match=0,
                        exec_info=exec_info,
                    )
                )
                continue
            actual = res.pop("output")
            assert not res
            if isinstance(actual.image, types.MaskedImage):
                mask = actual.image._mask
            else:
                mask = True
            kw = dict(output=actual, exec_info=exec_info)
            if io.output is not None:
                if actual.shape != io.output.shape:
                    kw.update(full_match=False, cell_match=0)
                else:
                    match = (actual.image._data == io.output.image._data) & mask
                    kw.update(
                        full_match=bool(np.all(match)), cell_match=float(np.mean(match))
                    )
            splits[split].append(ExampleEval(**kw))
    allv = list(itertools.chain(*splits.values()))
    return ChallengeEval(
        **meta,
        exec_info=exec_info,
        **{f"{k}_eval": tuple(v) for k, v in splits.items()},
        full_match=all(v.full_match for v in allv if v.full_match is not None),
        example_match=float(
            np.mean([v.full_match for v in allv if v.full_match is not None])
        ),
        cell_match=float(
            np.mean([v.cell_match for v in allv if v.cell_match is not None])
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path)
    parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args()

    with open(args.input, "rb") as fh:
        params = msgpack.unpack(fh)
    params = deserialise(params)

    result = _evaluate_solution(params["challenge"], params["solution"])

    result = serialise(result)

    with open(args.output, "wb") as fh:
        msgpack.pack(result, fh)
