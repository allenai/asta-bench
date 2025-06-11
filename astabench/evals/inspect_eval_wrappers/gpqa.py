import random

from inspect_ai import Task, task, task_with
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import choice

from astabench.evals.utils import format_multichoice_as_textin, mark_choices_and_score


@task
def gpqa_diamond(seed: int = 0) -> Task:
    try:
        from inspect_evals.gpqa import gpqa_diamond as inspect_gpqa
    except ImportError:
        raise ImportError("gpqa_diamond is not available")
    base_task = inspect_gpqa()

    dataset = []
    rng = random.Random(seed)
    for sample in base_task.dataset:
        sample = format_multichoice_as_textin(sample, shuffle=True, rng=rng)
        dataset.append(sample)

    return task_with(
        base_task,
        dataset=MemoryDataset(dataset),
        epochs=1,
        scorer=mark_choices_and_score(inner_scorer=choice()),
    )
