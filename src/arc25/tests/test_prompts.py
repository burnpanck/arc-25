from pathlib import Path

import pytest

from ..dataset import Dataset, SolutionDB
from ..prompts import PromptEncoder, parse_explanation


@pytest.mark.asyncio
async def test_solutions():
    data_dir = (Path(__file__).parent / "../../../data").resolve()
    repack_dir = data_dir / "repack"
    ds = await Dataset.from_binary(repack_dir / "all-challenges.cbor.xz")
    solutions = await SolutionDB.load(data_dir / "solutions")

    enc = PromptEncoder()

    for ckey, sol in solutions.solutions.items():
        chal = ds.challenges[ckey]
        resp = parse_explanation(sol)
        enc.prepare_chat_messages(chal, resp)
