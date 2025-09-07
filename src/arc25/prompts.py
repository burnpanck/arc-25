import itertools
import re
import typing
from types import SimpleNamespace

import attrs
import numpy as np

from .dataset import Challenge, ReasonedSolution
from .dsl.types import AnyImage, Color, Image, IOPair
from .facts import FactDefinition, default_facts

default_system_msg = (
    "You are a careful, code-generating ARC challenge solver assistant."
)
alterante_system_msg = """
You are an ARC reasoning-and-coding agent.
You must:
1) Infer the hidden rule from few visual I/O examples.
2) Write Python that uses the provided ARC-DSL to transform inputs to outputs.
Do not invent libraries; only use the ARC-DSL, Python stdlib or the available libs.
""".strip()

default_task_descr = """
You are given a number of example pairs of input and output grids
(plus a few examples without known output grids).
You must identify the underlying rule transforming input into output,
and write a python implementation for that rule.
Analyse the challenge carefully and in a structured way by:
 1.) Formulate a *description* of the relevant semantic entities in the inputs.
 2.) Describe the underlying *rule* in natural language or pseudo-code.
 3.) Create a *plan* on how to implement the rule in python.
 4.) Output an *implementation* in python.
""".strip()

C = Color
single_char_color_codes = {
    C.BLACK: "k",
    C.BLUE: "b",
    C.BROWN: "n",
    C.CYAN: "c",
    C.GRAY: "h",
    C.GREEN: "g",
    C.MAGENTA: "m",
    C.ORANGE: "o",
    C.RED: "r",
    C.YELLOW: "y",
}
del C


@attrs.frozen(kw_only=True)
class PromptEncoder:
    system_msg: str = default_system_msg
    task_descr: str = default_task_descr
    replace_all_colours: bool = False
    with_transpose: bool = False
    add_code_fence: bool = True

    # interesting parentheses: «»‹›〔〕【】〖〗❪❫❲❳❬❭❨❩⟨⟩
    colour_tokens: tuple[str, ...] = tuple(
        f"❲{single_char_color_codes[c]}❳" for c in Color
    )
    missing_colour_token: str = "❲ ❳"
    open_tokens: SimpleNamespace = SimpleNamespace(
        **{
            k: f"<{k}>"
            for k in [
                "demo",  # few-shot example wrapper
                "query",  # actual query in contrast to few-shot
                "input",  # challenge input (I/O pairs)
                "example",  # I/O pair (or just I)
                "grid",  # direct-encoded grid
                "facts",  # programmatically derived facts
                "descr",  # CoT of various aspects
                "rule",  # CoT of the actual rule
                "plan",  # CoT of the implementation plan
                "impl",  # python implementation
            ]
        }
    )
    close_tokens: SimpleNamespace = SimpleNamespace(
        **{k: f"</{k}>" for k in vars(open_tokens)}
    )

    @property
    def all_special_tokens(self) -> typing.Iterable[str]:
        yield from self.colour_tokens
        yield self.missing_colour_token
        yield from vars(self.open_tokens).values()
        yield from vars(self.close_tokens).values()

    fact_definitions: tuple[FactDefinition, ...] = default_facts

    def encode_grid(self, grid: AnyImage) -> str:
        h, w = grid.shape
        data = grid._data
        if isinstance(grid, Image):
            mask = np.ones(data.shape, bool)
        else:
            mask = data._mask
        egrids = {
            k: "\n".join(
                "".join(
                    self.colour_tokens[dv] if mv else self.missing_colour_token
                    for dv, mv in zip(dr, mr)
                )
                for (dr, mr) in zip(*[d.T if do_T else d for d in [data, mask]])
            )
            for k, do_T in (
                dict(rows=False, cols=True) if self.with_transpose else dict(rows=False)
            ).items()
        }
        o = self.open_tokens
        c = self.close_tokens
        body = "\n---\n".join(f"{k}:\n{v}" for k, v in egrids.items())
        return f"""
{o.grid}{h}×{w}
{body}
{c.grid}
        """.strip()

    def encode_example(self, example: IOPair) -> str:
        o = self.open_tokens
        c = self.close_tokens
        body = "\n".join(
            f"{k}:{self.encode_grid(v)}"
            for k in ["input", "output"]
            if (v := getattr(example, k)) is not None
        )
        return f"""
{o.example}
{body}
{c.example}
        """.strip()

    def encode_inputs(self, challenge: Challenge) -> str:
        o = self.open_tokens
        c = self.close_tokens
        body = "\n".join(
            f"{k}:\n" + "\n".join(self.encode_example(e) for e in v)
            for k in ["train", "test"]
            if (v := getattr(challenge, k))
        )
        return f"""
{o.input}
{body}
{c.input}
        """.strip()

    def encode_facts(self, challenge: Challenge) -> str:
        o = self.open_tokens
        c = self.close_tokens
        # TODO: handle colour replacement!
        body = "\n".join(
            f"- {descr}" for fd in self.fact_definitions if (descr := fd(challenge))
        )
        return f"""
{o.facts}
{body}
{c.facts}
        """.strip()

    def encode_prompt(self, challenge: Challenge) -> dict[str, str]:
        user_msg = f"""
{self.task_descr}
{self.encode_inputs(challenge)}
{self.encode_facts(challenge)}
""".strip()
        return SimpleNamespace(
            system=self.system_msg,
            user=user_msg,
        )

    def encode_response(self, response: ReasonedSolution) -> str:
        o = self.open_tokens
        c = self.close_tokens
        ret = []
        # TODO: handle colour replacement!
        assert not self.replace_all_colours
        for tok, cat, content in itertools.chain(
            [("descr", k, v) for k, v in response.descr.items()],
            [
                (k, "", v)
                for k, v in dict(
                    rule=response.rule_descr,
                    plan=response.impl_plan_descr,
                    impl=response.rule_impl,
                ).items()
            ],
        ):
            if content is None or not content.strip():
                continue
            content = content.strip()
            if tok == "impl":
                if self.add_code_fence:
                    assert not cat
                    cat = "```python"
                    content += "\n```"
            else:
                content += "\n"

            ret.append(
                f"""
{vars(o)[tok]}{cat}
{content}{vars(c)[tok]}
            """.strip()
            )
        return "\n".join(ret)

    def prepare_chat_messages(
        self, chal: Challenge, resp: ReasonedSolution | None = None
    ) -> dict:
        messages = vars(self.encode_prompt(chal))
        if resp is not None:
            messages.update(assistant=self.encode_response(resp))
        return messages


_header_rex = re.compile(r"\*\*(\w+):\*\*")


def parse_explanation(sol):
    resp = {}
    descr = {}
    prev = resp.setdefault(None, []), 0
    for m in _header_rex.finditer(sol.explanation):
        content, s = prev
        e = m.start()
        content.append(sol.explanation[s:e].strip())
        t = m.group(1).lower()
        t = dict(output="outputs", input="inputs").get(t, t)
        if t in {"hypothesis", "inputs", "outputs"}:
            content = descr.setdefault(t, [])
        else:
            c = {
                "rule": "rule_descr",
                "plan": "impl_plan_descr",
            }.get(t, None)
            if c is None:
                print(f"Unknown header: {t!r}")
            content = resp.setdefault(c, [])
        prev = content, m.end()
    content, s = prev
    e = len(sol.explanation)
    content.append(sol.explanation[s:e].strip())

    resp = {k: "".join(v) for k, v in resp.items() if "".join(v)}
    descr = {k: "".join(v) for k, v in descr.items() if "".join(v)}

    default = resp.pop(None, None)
    if default is not None:
        if "rule_descr" not in resp:
            resp["rule_descr"] = default
        else:
            raise ValueError(
                f"Have non-trivial description without known header:\n{default!r}"
            )

    return ReasonedSolution(
        **resp,
        descr=descr,
        rule_impl=sol.rule.strip(),
    )


def parse_larc_annotations(
    larc: dict[str, typing.Any],
) -> typing.Iterable[ReasonedSolution]:
    """Parses the LARC data for a single challenge,
    yielding reasoning for each describer who succeeded in solving the task,
    as-well as in explaining the task so that at least one builder succeeded.
    """
    for _describer_id, d in larc.items():
        if not d["succeeded_verification"]:
            continue
        suc = True
        for _builder_id, dd in d["builds"].items():
            suc = suc and dd["success"]
        if not suc:
            continue
        kw = {}
        for k, kk in dict(
            see="input_descr", grid="rule_descr", do="rule_descr"
        ).items():
            v = d[f"{k}_description"]
            if k != "grid":
                v = v.split("...", 1)[1]
            kw.setdefault(kk, []).append(v)
        kw = {k: "\n".join(v) for k, v in kw.items()}
        descr = dict(inputs=kw.pop("input_descr"))
        yield ReasonedSolution(
            descr=descr,
            **kw,
        )
