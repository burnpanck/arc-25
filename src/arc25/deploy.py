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
import etils.epath


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


async def get_data_files_for_deployment(
    data_root: anyio.Path, *, with_models: bool = True
) -> list[anyio.Path]:
    """
    Returns list of data files to include in deployments.
    Single source of truth for what data gets deployed to Docker, Kaggle, etc.
    """
    files = [
        data_root / "README.md",
        data_root / "known-good-solutions.txt",
    ]

    # Challenge datasets from repack directory
    repack_root = data_root / "repack"
    for fn in [
        "all-challenges.cbor.xz",
        "larc-human.cbor.xz",
        "harc-rule-descr.cbor.xz",
        "re-arc.cbor.xz",
    ]:
        files.append(repack_root / fn)

    # Solution files
    ssrc = data_root / "solutions"
    async for fn in ssrc.glob("*.txt"):
        files.append(fn)
    async for fn in ssrc.glob("*.py"):
        files.append(fn)

    if with_models:
        import etils.epath

        # Pretrained models
        models_root = etils.epath.Path("gs://576e2361-arc-agi-2/checkpoints/")
        for chkp in [
            #           "20251023-1137-vertex-ai-mae-tiny-4xL4-chkp-006912.msgpack.xz",
            #           "20251025-1452-vertex-ai-mae-small-4xL4-chkp-004096.msgpack.xz",
            #            "20251030-2020-vertex-ai-mae-small-4xv6e-chkp-003072.msgpack.xz",
            "20251030-2020-vertex-ai-mae-small-4xv6e-chkp-007568-final.msgpack.xz",
            "20251031-1133-vertex-ai-arc-solver-small-4xv6e-chkp-000768.msgpack.xz",
            "20251101-1834-vertex-ai-arc-solver-small-4xv6e-chkp-001024.msgpack.xz",
            "20251101-1834-vertex-ai-arc-solver-small-4xv6e-chkp-002403-final.msgpack.xz",
        ]:
            files.append(models_root / f"{chkp.split('-chkp-')[0]}/{chkp}")

    return files


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
                    if isinstance(src, etils.epath.Path):
                        srcind = src
                        rel = Path("models") / src.name
                    else:
                        srcind = rel = src.relative_to(data_root)
                    dst = dst_root / rel
                    print(f"Prepare {dst.relative_to(tdir)} from {srcind}")
                    if not dst.parent.exists():
                        await anyio.Path(dst.parent).mkdir(parents=True, exist_ok=True)
                    if isinstance(src, etils.epath.Path):
                        await anyio.to_thread.run_sync(lambda: src.copy(dst))
                        assert dst.exists()
                        print(f"{dst.name}: {dst.stat().st_size/2**20:.1f} MB")
                    else:
                        await anyio.to_thread.run_sync(shutil.copy2, src, dst)

            return copy_fn

        atdir = anyio.Path(tdir)
        await (atdir / "dataset-metadata.json").write_text(
            json.dumps(dataset.as_json())
        )

        # Get files to deploy
        files_to_copy = await get_data_files_for_deployment(data_root)

        async with anyio.create_task_group() as tg:
            for file in files_to_copy:
                tg.start_soon(make_copy_fn(file))
        await anyio.to_thread.run_sync(
            lambda: api.dataset_create_version(
                str(tdir), version_notes=msg, dir_mode="zip"
            )
        )


@dataset.command()
@click.argument("build-context", type=click.Path(path_type=Path))
async def prepare_docker_context(build_context: Path):
    """
    Copy data and notebook files to Docker build context.
    Single source of truth for what data gets deployed.
    """
    proj_root = await anyio.Path(__file__).parents[2].resolve()
    data_root = anyio.Path(proj_root / "data")
    notebooks_root = anyio.Path(proj_root / "notebooks")
    build_context = anyio.Path(build_context)

    # Ensure build context exists
    await build_context.mkdir(parents=True, exist_ok=True)

    lim = anyio.CapacityLimiter(total_tokens=8)

    async def copy_file(src: anyio.Path, dst: anyio.Path):
        async with lim:
            print(
                f"Copy {src.relative_to(proj_root)} -> {dst.relative_to(build_context)}"
            )
            await dst.parent.mkdir(parents=True, exist_ok=True)
            await anyio.to_thread.run_sync(shutil.copy2, src, dst)

    # Get data files
    data_files = await get_data_files_for_deployment(data_root, with_models=False)

    async with anyio.create_task_group() as tg:
        # Copy data files
        for src in data_files:
            rel = src.relative_to(proj_root)
            dst = build_context / rel
            tg.start_soon(copy_file, src, dst)

        # Copy all notebooks
        async for nb in notebooks_root.glob("*.ipynb"):
            rel = nb.relative_to(proj_root)
            dst = build_context / rel
            tg.start_soon(copy_file, nb, dst)

    print(f"Prepared Docker build context at {build_context}")


if __name__ == "__main__":
    dataset()
