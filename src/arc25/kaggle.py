import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import anyio
import asyncclick as click
import attrs


@attrs.frozen
class KaggleDatasetMeta:
    title: str
    id: str
    private: bool = True
    license: str = "unknown"
    subtitle: str | None = None
    description: str | None = None

    def as_json(self):
        ret = attrs.asdict(self, filter=lambda a, v: v is not None)
        ret.update(
            isPrivate=ret.pop("private"),
            licenses=[dict(name=ret.pop("license"))],
        )
        return ret


@click.group()
def dataset():
    pass


@dataset.command()
@click.option("-m", "--msg", default=None, help="Commit message")
async def update_training_data(msg: str):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    cfg = anyio.Path("~/.kaggle/kaggle.json")
    cfg = await cfg.expanduser()
    cfg = await cfg.resolve()
    if await cfg.exists():
        cfg = await cfg.expanduser()
        cfg = await cfg.resolve()
        cfg = json.loads(await cfg.read_text())
        username = cfg["username"]
    else:
        username = os.environ["KAGGLE_USERNAME"]

    proj_root = await anyio.Path(__file__).parents[2].resolve()
    data_root = proj_root / "data"
    repack_root = data_root / "repack"
    dataset = KaggleDatasetMeta(
        title="ARC Prize 2025 training data",
        id=f"{username}/arc25-training-data",
        subtitle="",
    )
    lim = anyio.CapacityLimiter(total_tokens=8)

    async with contextlib.AsyncExitStack() as stack:
        tdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        def make_copy_fn(src, dst_root=tdir):
            async def copy_fn():
                async with lim:
                    rel = src.relative_to(data_root)
                    dst = dst_root / rel
                    if dst_root != tdir:
                        print(f"Copy {rel} to {dst.relative_to(tdir)}")
                    else:
                        print(f"Copy {rel}")
                    if not dst.parent.exists():
                        await anyio.Path(dst.parent).mkdir(parents=True, exist_ok=True)
                    await anyio.to_thread.run_sync(shutil.copy2, src, dst)

            return copy_fn

        atdir = anyio.Path(tdir)
        await (atdir / "dataset-metadata.json").write_text(
            json.dumps(dataset.as_json())
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(make_copy_fn(data_root / "README.md"))
            for fn in [
                "all-challenges.cbor.xz",
                "larc-human.cbor.xz",
                "harc-rule-descr.cbor.xz",
                "re-arc.cbor.xz",
            ]:
                tg.start_soon(make_copy_fn(repack_root / fn))
            tg.start_soon(make_copy_fn(data_root / "known-good-solutions.txt"))
            ssrc = data_root / "solutions"
            async for fn in ssrc.glob("*.txt"):
                tg.start_soon(make_copy_fn(fn))
            async for fn in ssrc.glob("*.py"):
                tg.start_soon(make_copy_fn(fn))
        await anyio.to_thread.run_sync(
            lambda: api.dataset_create_version(
                str(tdir), version_notes=msg, dir_mode="zip"
            )
        )


if __name__ == "__main__":
    dataset()
