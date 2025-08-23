import argparse
import contextlib
import importlib.metadata
import logging
import sys
from pathlib import Path

import attrs
import nicegui
import platformdirs
import yaml
from nicegui import ui


@attrs.mutable
class App:
    pass


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

    for log in """
        asyncio
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

    logger = logging.getLogger("arc25.explorer")
    logger.info(
        f"arc25-explorer v{importlib.metadata.version("yde-arc25")} is starting"
    )
    logger.debug(f"Default location of app-config.yaml: {apd.user_config_dir}")
    logger.debug(f"Location of logs: {apd.user_log_dir}")

    _orig_lifespan_context = nicegui.app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(nicegui_app):
        async with _orig_lifespan_context(nicegui_app):
            async with App.run() as app:
                nicegui_app.app_impl = app
                yield

    nicegui.app.router.lifespan_context = lifespan

    #    kw.setdefault("storage_secret", "1234123asdfqsd")

    if args.on_air:
        kw["on_air"] = args.on_air

    ui.run(**kw)


if __name__ in {"__main__", "__mp_main__"}:
    run()
