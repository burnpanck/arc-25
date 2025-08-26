import argparse
import contextlib
import importlib.metadata
import logging
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import attrs
import nicegui
import platformdirs
import yaml
from fastapi import Request
from nicegui import ui

from .. import tools as arc_tools
from ..dataset import Dataset, Solution, SolutionDB

logger = logging.getLogger("arc25.explorer")


@attrs.mutable
class App:
    datasets: dict[str, Dataset]
    solutions_db: SolutionDB
    _pending_solutions: dict[str, Solution] = attrs.field(factory=dict)
    _new_solution: anyio.Event = attrs.field(factory=anyio.Event)

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
        async with contextlib.AsyncExitStack() as stack:
            tg = await stack.enter_async_context(anyio.create_task_group())
            stack.callback(tg.cancel_scope.cancel)
            tg.start_soon(self._db_runner)
            yield self

    def store_solution(self, sol: Solution):
        self._pending_solutions[sol.id] = sol
        self._new_solution.set()

    async def _db_runner(self):
        async def single_pass():
            pending = self._pending_solutions
            for id in list(pending):
                logger.debug(f"Storing solution for challenge {id}")
                sol = pending.pop(id)
                try:
                    await self.solutions_db.store(sol)
                except BaseException:
                    pending.setdefault(id, sol)
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

rule_placeholder = """
def solution(input: Canvas) -> Canvas:
    # determine output from input
    return output
""".lstrip()


@ui.page("/")
def main_page(*, request: Request):
    app: App = request.app.app_impl

    fig = ui.matplotlib().figure

    def update_figure(cur_c):
        chal = cur_ds.challenges[cur_c]
        with fig:
            fig.clear()
            arc_tools.show_test_case(
                chal.train + chal.test,
                n_train=len(chal.train),
                fig=fig,
                example_width=3,
                orientation="v",
            )

    with ui.left_drawer(fixed=True).props("width=500"):
        initial_value = "training"
        ds_select = ui.select(
            {d.id: d.title for d in app.datasets.values()},
            value=initial_value,
        )
        if False:

            def load_dataset(ds_id):
                ds = app.datasets[ds_id]
                return [dict(id=k, status="") for k, v in ds.challenges.items()]

            ds_table = ui.table(
                columns={k: k.title() for k in ["id", "status"]},
                rows=load_dataset(initial_value)[:20],
                row_key="id",
                #            selection="single",
            )

            def update_dataset(evt):
                ds = load_dataset(evt.value)
                ds_table.clear()
                ds_table.add_rows(ds)

            ds_select.on_value_change(update_dataset)
        cur_ds = app.datasets[initial_value]
        ckeys = list(cur_ds.challenges)
        csel = ui.slider(min=0, max=len(ckeys) - 1, value=0)
        cur_c = ckeys[csel.value]

        def update_dataset(evt):
            nonlocal cur_ds, ckeys
            cur_ds = app.datasets[evt.value]
            ckeys = list(cur_ds.challenges)
            csel.set_value(0)
            csel.props["max"] = len(ckeys) - 1
            update_challenge(SimpleNamespace(value=0))

        ds_select.on_value_change(update_dataset)
        clabel = ui.label(f"{cur_ds.title}: {1}/{len(ckeys)}")

        store_holdoff = anyio.current_time()
        def update_challenge(evt):
            nonlocal cur_c, store_holdoff
            remember_solution()
            store_holdoff = anyio.current_time()
            idx = int(evt.value)
            cur_c = ckeys[idx]
            clabel.set_text(f"{cur_ds.title}: {cur_c} ({idx+1}/{len(ckeys)})")
            sol = app.solutions_db.solutions.get(cur_c, Solution.make(id=cur_c))
            explanation.set_value(sol.explanation)
            r = sol.rule
            if not r.strip():
                r = rule_placeholder
            rule.set_value(r)
            update_figure(cur_c)

        csel.on_value_change(update_challenge)
        with ui.row():
            prev = ui.button("Prev")
            prev.on_click(lambda: csel.set_value(max(0, csel.value - 1)))
            next = ui.button("Next")
            next.on_click(lambda: csel.set_value(min(len(ckeys) - 1, csel.value + 1)))

        explanation = ui.textarea(
            placeholder="Explanation",
        ).classes("w-full")
        rule = ui.codemirror(
            language="python",
            theme="githubLight",
        ).classes("w-full h-full")

        def remember_solution():
            r = rule.value.strip()
            if r == rule_placeholder.strip():
                r = ""
            sol = Solution(
                id=cur_c,
                explanation=explanation.value.strip()+"\n",
                rule=r+"\n",
            )
            if not sol.is_empty and sol != app.solutions_db.solutions.get(sol.id):
                if anyio.current_time() < store_holdoff+1:
                    logger.warning(f"Not storing solution {sol.id} due to being recently loaded: {sol}")
                    return
                logger.debug(f"Storing solution {sol.id}: {sol}")
                app.store_solution(sol)

        explanation.on_value_change(lambda evt: remember_solution())
        rule.on_value_change(lambda evt: remember_solution())
        update_challenge(SimpleNamespace(value=0))

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
        f"arc25-explorer v{importlib.metadata.version("yde-arc25")} is starting"
    )
    logger.debug(f"Default location of app-config.yaml: {apd.user_config_dir}")
    logger.debug(f"Location of logs: {apd.user_log_dir}")

    proj_root = Path(__file__).parents[3].resolve()
    data_path = proj_root / "data"
    db_root = data_path / "solutions"
    db_root.mkdir(exist_ok=True, parents=True)
    challenges_root = data_path / "arc-prize-2025.zip"

    async def load_datasets():
        logger.info(f"Loading data from {challenges_root}")
        datasets = {}
        for k in ["training", "evaluation", "test"]:
            ds = await Dataset.load(
                id=k,
                root=challenges_root,
                challenges=f"arc-agi_{k}_challenges.json",
                solutions=f"arc-agi_{k}_solutions.json" if k != "test" else None,
            )
            datasets[k] = ds
        datasets["combined"] = Dataset(
            id="combined",
            challenges=dict(
                itertools.chain(*[ds.challenges.items() for ds in datasets.values()])
            ),
        )
        return datasets

    _orig_lifespan_context = nicegui.app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(nicegui_app):
        async with _orig_lifespan_context(nicegui_app):
            datasets = await load_datasets()
            async with App.run(
                datasets=datasets,
                solutions_db=db_root,
            ) as app:
                nicegui_app.app_impl = app
                yield

    nicegui.app.router.lifespan_context = lifespan

    #    kw.setdefault("storage_secret", "1234123asdfqsd")

    if args.on_air:
        kw["on_air"] = args.on_air

    kw.setdefault("uvicorn_reload_dirs", "src/")

    ui.run(**kw)


if __name__ in {"__main__", "__mp_main__"}:
    run()
