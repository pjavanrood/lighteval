"""
name:
Qasper

dataset:
allenai/qasper

abstract:
QASPER is a dataset for question answering on scientific research papers. It
consists of 5,049 questions over 1,585 Natural Language Processing papers. Each
question is written by an NLP practitioner who read only the title and abstract
of the corresponding paper, and the question seeks information present in the
full text. The questions are then answered by a separate set of NLP
practitioners who also provide supporting evidence to answers.

languages:
english

tags:
qa, scientific

paper:
https://arxiv.org/abs/2105.03011
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def qasper_prompt(line, task_name: str = None):
    """Handle the parquet format which has nested qas structure."""
    results = []
    title = line.get("title", "")
    # Use abstract as the passage since parquet format doesn't have 'passage' field
    passage = line.get("abstract", "")

    qas = line.get("qas", {})
    questions = qas.get("question", [])
    answers_list = qas.get("answers", [])

    for i, question in enumerate(questions):
        gold = ""
        if i < len(answers_list):
            answer_entry = answers_list[i]
            if isinstance(answer_entry, dict) and "answer" in answer_entry:
                annotator_answers = answer_entry["answer"]
                if annotator_answers and len(annotator_answers) > 0:
                    first_answer = annotator_answers[0]
                    if first_answer.get("free_form_answer"):
                        gold = first_answer["free_form_answer"]
                    elif first_answer.get("extractive_spans"):
                        gold = " ".join(first_answer["extractive_spans"])
                    elif first_answer.get("yes_no") is not None:
                        gold = "yes" if first_answer["yes_no"] else "no"

        if gold:
            results.append(Doc(
                task_name=task_name,
                query=f"Title: {title}\n\nPassage: {passage}\n\nQuestion: {question}\nAnswer: ",
                gold_index=0,
                choices=[gold],
            ))

    # Return results if we found any questions with answers, otherwise return a single Doc
    return results if results else Doc(
        task_name=task_name,
        query=f"Title: {title}\n\nPassage: {passage}\n\nQuestion: {questions[0] if questions else ''}\nAnswer: ",
        gold_index=0,
        choices=[""],
    )


qasper = LightevalTaskConfig(
    name="qasper",
    prompt_function=qasper_prompt,
    hf_repo="allenai/qasper",
    hf_revision="refs/convert/parquet",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.f1_score],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    qasper,
]
