import os
from pathlib import Path

import anyio
import pytest

from arc25.dataset import Dataset, SolutionDB
from arc25.sandbox import evaluate_solution


def available_cpus():
    if hasattr(os, "sched_getaffinity"):  # Linux/Unix
        return len(os.sched_getaffinity(0))
    n = os.cpu_count()  # Fallback (system-wide)
    return n if n is not None else 1


@pytest.mark.asyncio
async def test_all_solutions():
    data_dir = (Path(__file__).parent / "../../../data").resolve()
    ds = await Dataset.from_binary(data_dir / "all-challenges.cbor.xz")
    solutions = await SolutionDB.load(data_dir / "solutions")
    known_good_path = anyio.Path(data_dir / "known-good-solutions.txt")
    known_good = await known_good_path.read_text()
    known_good = frozenset(known_good.split())
    good = set()
    bad = set()

    lim = anyio.CapacityLimiter(total_tokens=available_cpus())

    async def handle(id, sol):
        async with lim:
            chal = ds.challenges[id]
            eval = await evaluate_solution(chal, sol)
            (good if eval.full_match else bad).add(id)
            if id in known_good and not eval.full_match:
                print(f"Solution to challenge {id} should not fail:")
                print("\n".join(eval.summary(with_ansi=True)))

    async with anyio.create_task_group() as tg:
        for id, sol in solutions.solutions.items():
            if not sol.rule.strip():
                continue
            tg.start_soon(handle, id, sol)
    now_available = good | bad
    still_good = known_good & good
    newly_bad = known_good & bad
    newly_missing = known_good - now_available
    extra_good = good - known_good
    extra_bad = bad - known_good
    if extra_good:
        print(f"Have new good solutions: {sorted(extra_good)}")
        await known_good_path.write_text(
            "".join(f"{k}\n" for k in sorted(known_good | extra_good))
        )
    if still_good != known_good:
        raise ValueError(
            f"We lost some solutions! Newly bad: {sorted(newly_bad)},"
            f" missing: {sorted(newly_missing)}."
            f" (Unfinished: {sorted(extra_bad)})"
        )
    if extra_bad:
        print(f"Have unfinished solutions: {sorted(extra_bad)}")
