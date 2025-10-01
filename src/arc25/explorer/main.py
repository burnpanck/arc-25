import argparse
import contextlib
import importlib.metadata
import itertools
import logging
import random
import re
import sys
import typing
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakKeyDictionary

import anyio
import attrs
import nicegui
import platformdirs
import yaml
from fastapi import Request
from nicegui import ui

from .. import tools as arc_tools
from ..dataset import Dataset, Solution, SolutionDB, load_datasets
from ..sandbox import ChallengeEval, evaluate_solution

logger = logging.getLogger("arc25.explorer")


@attrs.mutable
class App:
    dataset: Dataset
    solutions_db: SolutionDB
    solutions: dict[str, Solution] = attrs.field(factory=dict)
    _pending_solutions: set[str] = attrs.field(factory=set)
    _new_solution: anyio.Event = attrs.field(factory=anyio.Event)

    _pending_evaluations: dict[Solution, WeakKeyDictionary] = attrs.field(factory=dict)
    _need_evaluation: anyio.Event = attrs.field(factory=anyio.Event)
    evaluations: dict[str, ChallengeEval] = attrs.field(factory=dict)

    @classmethod
    @contextlib.asynccontextmanager
    async def run(
        cls,
        solutions_db: Path,
        **kw,
    ):
        self = cls(
            solutions_db=await SolutionDB.load(solutions_db),
            **kw,
        )
        self.solutions.update(self.solutions_db.solutions)
        async with contextlib.AsyncExitStack() as stack:
            tg = await stack.enter_async_context(anyio.create_task_group())
            stack.callback(tg.cancel_scope.cancel)
            tg.start_soon(self._db_runner)
            tg.start_soon(self._eval_runner)
            yield self

    def store_solution(self, sol: Solution):
        id = sol.id
        prev = self.solutions.get(id)
        self.solutions[id] = sol
        if prev is not None and prev.rule != sol.rule:
            self.evaluations.pop(id, None)
        self._pending_solutions.add(sol.id)
        self._new_solution.set()

    async def _db_runner(self):
        async def single_pass():
            pending = self._pending_solutions
            for id in list(pending):
                pending.discard(id)
                # logger.debug(f"Storing solution for challenge {id}")
                sol = self.solutions[id]
                try:
                    await self.solutions_db.store(sol)
                except BaseException:
                    pending.add(id)
                    raise

        try:
            while True:
                await self._new_solution.wait()
                self._new_solution = anyio.Event()
                await single_pass()
                await anyio.sleep(1)
        finally:
            with anyio.CancelScope(shield=True):
                await single_pass()

    def evaluate_solution(
        self, sol: Solution, owner: typing.Any, callback: typing.Callable
    ):
        logger.debug(f"Trigger evaluation of solution for {sol.id}")
        self._pending_evaluations.setdefault(sol, WeakKeyDictionary())[owner] = callback
        self._need_evaluation.set()

    async def _eval_runner(self):
        async def single_pass():
            pending = self._pending_evaluations
            for sol in list(pending):
                clients = pending.pop(sol)
                if not clients:
                    continue
                id = sol.id
                if sol.rule != self.solutions[id].rule:
                    logger.info(
                        f"Skipping evaluation of outdated solution for challenge {id}"
                    )
                    continue
                logger.debug(f"Evaluating solution for challenge {id}")
                chal = self.dataset.challenges[id]
                eval = await evaluate_solution(
                    challenge=chal,
                    solution=sol,
                )
                if sol.rule != self.solutions[id].rule:
                    logger.info(
                        f"Ignoring evaluation of outdated solution for challenge {id}"
                    )
                    continue
                self.evaluations[sol.id] = eval
                for owner, callback in clients.items():
                    try:
                        callback(owner, sol, eval)
                    except Exception as ex:
                        logger.warning(
                            f"Ignoring exception in evaluation callback: {ex!r}"
                        )

        try:
            while True:
                await self._need_evaluation.wait()
                self._need_evaluation = anyio.Event()
                await single_pass()
                await anyio.sleep(1)
        finally:
            with anyio.CancelScope(shield=True):
                await single_pass()


rule_placeholder = """
def solution(input: Image) -> Image:
    # determine output from input
    return output
""".lstrip()


@ui.page("/")
def main_page(*, request: Request):
    nicegui_app = request.app
    app: App = nicegui_app.app_impl

    fig = ui.matplotlib().figure

    def update_figure(cur_c):
        chal = app.dataset.challenges[cur_c]
        eval = app.evaluations.get(cur_c)
        if eval is None:
            triples = chal.get_empty_eval_triples()
        else:
            triples = eval.get_eval_triples()
            show_eval_output(eval)
        with fig:
            fig.clear()
            arc_tools.show_test_case(
                triples,
                n_train=len(chal.train),
                fig=fig,
                example_width=3,
                orientation="v",
            )

    with ui.left_drawer(fixed=True).props("width=600"):
        ds_select = ui.select(
            {d: d for d in app.dataset.subsets},
        ).bind_value(nicegui_app.storage.user, "dataset")

        cur_ds = ds_select.value or sorted(app.dataset.subsets)[0]
        ckeys = tuple(sorted(app.dataset.subsets[cur_ds]))
        csel = ui.slider(min=0, max=len(ckeys) - 1).bind_value(
            nicegui_app.storage.user, "challenge"
        )
        cur_c = ckeys[min(csel.value or 0, len(ckeys) - 1)]

        def update_dataset(evt):
            nonlocal cur_ds, ckeys
            cur_ds = evt.value
            ckeys = tuple(sorted(app.dataset.subsets[cur_ds]))
            csel.set_value(0)
            csel.props["max"] = len(ckeys) - 1
            update_challenge(SimpleNamespace(value=0))

        ds_select.on_value_change(update_dataset)
        clabel = ui.label(f"{cur_ds}: {1}/{len(ckeys)}")

        store_holdoff = anyio.current_time()

        def eval_callback(owner, sol, eval):
            logger.debug(
                f"Eval for solution for {sol.id} completed: {eval.full_match=}"
            )
            try:
                update_figure(sol.id)
            except Exception:
                import traceback

                traceback.print_exc()
                raise

        def update_challenge(evt):
            nonlocal cur_c, store_holdoff
            remember_solution()
            store_holdoff = anyio.current_time()
            idx = min(int(evt.value or 0), len(ckeys) - 1)
            cur_c = ckeys[idx]
            clabel.set_text(f"{cur_ds}: {cur_c} ({idx+1}/{len(ckeys)})")
            sol = app.solutions.get(cur_c, Solution.make(id=cur_c))
            explanation.set_value(sol.explanation)
            r = sol.rule
            if not r.strip():
                r = rule_placeholder
            rule.set_value(r)
            output.clear()
            if not sol.is_empty and cur_c not in app.evaluations:
                app.evaluate_solution(sol, fig, eval_callback)
            update_figure(cur_c)

        csel.on_value_change(update_challenge)
        with ui.row():
            prev = ui.button("Prev")
            prev.on_click(lambda: csel.set_value(max(0, csel.value - 1)))
            next = ui.button("Next")
            next.on_click(lambda: csel.set_value(min(len(ckeys) - 1, csel.value + 1)))
            rnd = ui.button("Random")
            rnd.on_click(lambda: csel.set_value(random.randint(0, len(ckeys) - 1)))

        explanation = ui.textarea(
            placeholder="Explanation",
        ).classes("w-full")
        rule = ui.codemirror(
            language="python",
            theme="githubLight",
        ).classes("w-full h-full")
        output = ui.log().classes("w-full h-full")

        def show_eval_output(eval: ChallengeEval):
            output.clear()
            if eval is None:
                return
            ansi_re = re.compile("\x1b" r"\[([0-9;]+)m(.*)" "\x1b" r"\[0m$", re.DOTALL)
            ansi_map = {
                v: f"text-{k}"
                for k, v in dict(
                    red=31,
                    green=32,
                    # yellow=33,
                    orange=33,
                    blue=34,
                    magenta=35,
                    cyan=36,
                    white=37,
                    bold=1,
                ).items()
            }
            for line in eval.summary():
                classes = []
                if m := ansi_re.match(line):
                    line = m.group(2)
                    for k in m.group(1).split(";"):
                        classes.append(ansi_map[int(k)])
                output.push(line, classes=" ".join(classes))

        def remember_solution():
            r = rule.value.strip()
            if r == rule_placeholder.strip():
                r = ""
            sol = Solution(
                id=cur_c,
                explanation=explanation.value.strip() + "\n",
                rule=r + "\n",
            )
            if not sol.is_empty and sol != app.solutions.get(sol.id):
                if anyio.current_time() < store_holdoff + 1:
                    # logger.warning(f"Not storing solution {sol.id} due to being recently loaded: {sol}")
                    return
                # logger.debug(f"Storing solution {sol.id}: {sol}")
                app.store_solution(sol)
                if sol.id not in app.evaluations:
                    app.evaluate_solution(sol, fig, eval_callback)

        explanation.on_value_change(lambda evt: remember_solution())
        rule.on_value_change(lambda evt: remember_solution())
        update_challenge(SimpleNamespace(value=csel.value))

    update_figure(cur_c)


def run(**kw):
    parser = argparse.ArgumentParser(prog="arc25-explorer")
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--on-air", nargs="?", type=str, const=True, default=None)
    parser.add_argument(
        "--self-test", action="store_true", default=False, help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    if args.self_test:
        from ..tests import test_importing

        test_importing.test_importing()
        print("Self-test successful.")
        sys.exit(0)

    apd = platformdirs.PlatformDirs("arc25-explorer", "yde")

    if args.config:
        config_file = args.config
    else:
        config_file_cand = [
            d / "app-config.yaml"
            for d in [
                Path(".").resolve(),
                Path(apd.user_config_dir),
            ]
        ]
        for cand in config_file_cand:
            if cand.exists():
                config_file = cand
                break
        else:
            print(
                "No config file found. Candidate locations:\n"
                + "\n".join(str(p) for p in config_file_cand)
            )
    if False:
        with open(config_file, "rt") as fh:
            config = yaml.safe_load(fh)

        try:
            config = AppConfig.from_yaml(config, local_dirs=apd)  # noqa
        except Exception as ex:
            raise RuntimeError(f"Config file {config_file} is invalid") from ex

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    for log in """
        asyncio
        matplotlib
    """.split():
        if log.startswith("#"):
            continue
        log = logging.getLogger(log)
        log.setLevel(logging.INFO)
    for log in """
        watchfiles
    """.split():
        if log.startswith("#"):
            continue
        log = logging.getLogger(log)
        log.setLevel(logging.WARNING)

    logger.info(
        f"arc25-explorer v{importlib.metadata.version('yde-arc25')} is starting"
    )
    logger.debug(f"Default location of app-config.yaml: {apd.user_config_dir}")
    logger.debug(f"Location of logs: {apd.user_log_dir}")

    proj_root = Path(__file__).parents[3].resolve()
    data_path = proj_root / "data"
    db_root = data_path / "solutions"
    db_root.mkdir(exist_ok=True, parents=True)
    dataset_file = data_path / "repack/all-challenges.cbor.xz"

    _orig_lifespan_context = nicegui.app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(nicegui_app):
        async with _orig_lifespan_context(nicegui_app):
            ds = await Dataset.from_binary(dataset_file)
            async with App.run(
                dataset=ds,
                solutions_db=db_root,
            ) as app:
                nicegui_app.app_impl = app
                yield

    nicegui.app.router.lifespan_context = lifespan

    kw.setdefault("storage_secret", "27c245153ef0511d7a3b933107de24b3")

    if args.on_air:
        kw["on_air"] = args.on_air

    kw.setdefault("uvicorn_reload_dirs", "src/")

    ui.run(**kw)


if __name__ in {"__main__", "__mp_main__"}:
    run()
