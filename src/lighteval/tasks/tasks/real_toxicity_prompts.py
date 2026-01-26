"""
name:
Real Toxicity Prompts

dataset:
allenai/real-toxicity-prompts

abstract:
The RealToxicityPrompts dataset for measuring toxicity in prompted model generations

languages:
english

tags:
generation, safety

paper:
https://aclanthology.org/2020.findings-emnlp.301/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def real_toxicity_prompts_prompt(line, task_name: str = None):
    # Extract prompt text (nested under 'prompt' -> 'text')
    text = (
        line["prompt"]["text"]
        if isinstance(line.get("prompt"), dict) and "text" in line["prompt"]
        else line.get("text", "")
    )
    # Extract continuation text as the expected output for metrics
    continuation = (
        line["continuation"]["text"]
        if isinstance(line.get("continuation"), dict) and "text" in line["continuation"]
        else ""
    )
    return Doc(task_name=task_name, query=text, choices=[continuation], gold_index=0)


real_toxicity_prompts = LightevalTaskConfig(
    name="real_toxicity_prompts",
    prompt_function=real_toxicity_prompts_prompt,
    hf_repo="allenai/real-toxicity-prompts",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.f1_score],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    real_toxicity_prompts,
]
