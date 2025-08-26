from pathlib import Path

import anyio
import pytest

from arc25.dataset import Dataset, Solution, SolutionDB, load_datasets
from arc25.sandbox import evaluate_solution


@pytest.mark.asyncio
async def test_all_solutions():
    data_dir = (Path(__file__).parent / "../../../data").resolve()
    datasets = await load_datasets(data_dir / "arc-prize-2025.zip")
    solutions = await SolutionDB.load(data_dir / "solutions")
    known_good_path = anyio.Path(data_dir / "known-good-solutions.txt")
    known_good = await known_good_path.read_text()
    known_good = frozenset(known_good.split())
    ds = datasets["combined"]
    good = set()
    bad = set()
    for id, sol in solutions.solutions.items():
        if not sol.rule.strip():
            continue
        chal = ds.challenges[id]
        eval = await evaluate_solution(chal, sol)
        (good if eval.full_match else bad).add(id)
    now_available = good | bad
    still_good = known_good & good
    newly_bad = known_good & bad
    newly_missing = known_good - now_available
    extra_good = good - known_good
    extra_bad = bad - known_good
    if still_good != known_good:
        raise ValueError(
            f"We lost some solutions! Newly bad: {sorted(newly_bad)},"
            f" missing: {sorted(newly_missing)}."
            f" (Additional bad: {sorted(extra_bad)})"
        )
    if extra_good:
        print(f"Have new good solutions: {sorted(extra_good)}")
        await known_good_path.write_text(
            "".join(f"{k}\n" for k in sorted(known_good | extra_good))
        )
