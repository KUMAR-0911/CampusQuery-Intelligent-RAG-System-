
"""LangSmith evaluation for the CampusQuery RAG pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# Load environment variables before reading model settings.
load_dotenv(Path(__file__).parent / ".env")

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_groq import ChatGroq
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT

from config import DEFAULT_CONFIG
from RETRIVAL.pipeline import answer_question, get_vector_store


# ========================= TEST CASES =========================
TEST_CASES = [
    {
        "id": "case-001",
        "question": "Who heads the disciplinary committee (ISDC) at NIT Agartala, and who is the convenor?",
        "reference_answer": "The Dean (Student Welfare) is the Chairperson of the ISDC, and the Chief Warden is the Convenor.",
    },
    {
        "id": "case-002",
        "question": "Which section of the NIT Act 2007 gives the Director the power to maintain discipline?",
        "reference_answer": "Article 17(2) of the NIT Act 2007. In rare cases the Director may also invoke clause 17(2) to decide a student disciplinary issue on his own.",
    },
    {
        "id": "case-003",
        "question": "How much can I be fined for breaking a rule, and what happens if I damage college property?",
        "reference_answer": "The financial penalty can be up to Rs. 5,000/-. For damage to Institute property caused by irresponsible behavior, the cost is recovered at 3 times the replacement cost, in addition to other disciplinary measures.",
    },
    {
        "id": "case-004",
        "question": "If a disciplinary complaint is filed against me, how soon will a decision come and how long do I have to appeal?",
        "reference_answer": "The decision is made within 7 days of reporting, and the student has 7 days from the decision to appeal.",
    },
    {
        "id": "case-005",
        "question": "What are the steps in the disciplinary process?",
        "reference_answer": "The four steps are Report, Investigate, Decide, and Appeal.",
    },
    {
        "id": "case-006",
        "question": "What happens if I get caught cheating in an exam, and what if I do it again?",
        "reference_answer": "The student is awarded an 'F' grade in the subject(s) in which unfair means were adopted and is reported to a counseling committee. If the student repeats the offense, he/she will be rusticated from the institute.",
    },
    {
        "id": "case-007",
        "question": "I'm a fresher and got ragged but didn't tell anyone. Can I get punished too?",
        "reference_answer": "Yes. If a junior student yields to ragging and does not inform the institute or hostel authorities, or willfully withholds information in an investigation, it is treated as indiscipline and attracts punishments comparable to those for ragging. The suppression of a complaint by a junior does not exempt the senior from punishment.",
    },
    {
        "id": "case-008",
        "question": "What counts as a major punishment, and do I lose anything else because of it?",
        "reference_answer": "Major punishments are items (v) to (ix): partial or complete debarment from campus placement, academic suspension for one or two consecutive semesters, expulsion from the hostel, rustication for a specified period, and outright expulsion from the Institute. They also debar the student from all academic medals and prizes, as well as important non-academic awards.",
    },
    {
        "id": "case-009",
        "question": "Can boys stay out of the hostel at night? Also, how many days of mess leave can I take in a month?",
        "reference_answer": "Boys are not allowed to stay outside the hostel from 10:00 PM to 4:30 AM for any reason except medical issues. A boarder can take a maximum of 8 days of leave from the mess in a month, with approval from the hostel authority.",
    },
    {
        "id": "case-010",
        "question": "How many times can I come late to the hostel before I get a warning letter?",
        "reference_answer": "After 3 late entries, one warning letter is issued. If the student is still late despite the warning letter, it leads to disciplinary action.",
    },
    {
        "id": "case-011",
        "question": "Can my mother visit me at Gargi Hostel? What are the timings and rules?",
        "reference_answer": "Only lady guardians are allowed to visit the boarders, on Saturdays, Sundays and Institute holidays, from 12:30 PM to 3:30 PM, for 30 minutes (or as permitted by the concerned hostel authority), after submitting a photocopy of a government-issued identity proof.",
    },
    {
        "id": "case-012",
        "question": "If I get a P-3 punishment, what else gets added automatically?",
        "reference_answer": "P-1 (warning with undertaking for not repeating the mistake) and P-2 (parents of the student will be intimated) are automatically added to all punishments from P-3 onwards.",
    },
    {
        "id": "case-013",
        "question": "What is the punishment for sexually harassing a student, and for having a gun or explosives on campus?",
        "reference_answer": "Sexual harassment of a fellow student (OA-35) carries deterrent P15/P16. Possessing explosives, firearms or such dangerous items (OA-42) carries P16/P17.",
    },
    {
        "id": "case-014",
        "question": "What happens to someone who files a false complaint against another student?",
        "reference_answer": "The penalty on the student who made the false complaint will be at least one step higher than the highest penalty recommended for the offense charged.",
    },
    {
        "id": "case-015",
        "question": "Can the Director reduce or increase the punishment the disciplinary committee recommends?",
        "reference_answer": "The Director may approve the recommendation, return it to the ISDC for re-evaluation, reduce the punishment, or submit the matter to the Senate. The Director may make minor changes to the nature of the punishment or reduce its level/quantum, but shall not increase the punishment recommended by the ISDC.",
    },
    {
        "id": "case-016",
        "question": "A senior called me to meet him and I didn't report it. Will I be punished as a fresher?",
        "reference_answer": "Yes. Not reporting a ragging incident (OB-1 to OB-5) involving oneself is offense OC-2, which carries deterrent P1-P3.",
    },
    {
        "id": "case-017",
        "question": "How much is the B.Tech fee per semester at NIT Agartala?",
        "reference_answer": "This information is not available in the Student Discipline Manual. The system should say it cannot find the fee structure in the provided document.",
    },
    {
        "id": "case-018",
        "question": "What does NIT stand for?",
        "reference_answer": "NIT stands for National Institute of Technology.",
    },
    {
        "id": "case-019",
        "question": "Who is the current PM of India?",
        "reference_answer": "This is outside the scope of the Student Discipline Manual. The system should say the answer is not in the provided document.",
    },
    {
        "id": "case-020",
        "question": "What was the placement percentage at NIT Agartala in 2025?",
        "reference_answer": "This information is not available in the Student Discipline Manual. The manual only covers placement-related disciplinary offenses, not placement statistics. The system should say it cannot find this in the document.",
    },
]


# ========================= HELPER FUNCTIONS =========================

def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing {name} in .env")
    return value


def _is_rate_limited(error: BaseException) -> bool:
    current = error

    while current:
        response = getattr(current, "response", None)
        status = (
            getattr(current, "status_code", None)
            or getattr(response, "status_code", None)
        )

        if status == 429:
            return True

        current = current.__cause__ or current.__context__

    return False


def _retry_after(error: BaseException) -> float | None:
    current = error

    while current:
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")

        if value:
            try:
                return max(0.0, float(value))
            except (ValueError, TypeError):
                pass

        current = current.__cause__ or current.__context__

    return None


def _run_with_retries(
    label: str,
    operation: Callable[[], Any],
    delay_seconds: float,
    max_retries: int,
) -> Any:

    for attempt in range(max_retries + 1):
        if delay_seconds:
            time.sleep(delay_seconds)

        try:
            return operation()

        except Exception as error:
            if not _is_rate_limited(error) or attempt == max_retries:
                raise

            wait = _retry_after(error) or min(2 ** attempt, 30)

            print(
                f"[evaluation] {label}: Rate limited. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"{label} failed.")


def _get_score(result: dict[str, Any]) -> bool:
    score = result.get("score")

    if isinstance(score, bool):
        return score

    if isinstance(score, (int, float)):
        return score >= 0.5

    if isinstance(score, str):
        return score.strip().lower() in {
            "true", "pass", "yes", "1"
        }

    raise ValueError(f"Invalid evaluation score: {score!r}")


def _chunks_to_context(chunks: list[Any]) -> list[str]:
    """Convert retrieved chunks into text documents for OpenEvals."""
    documents = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("page_content")
        else:
            text = getattr(chunk, "page_content", None)

        if text:
            documents.append(str(text).strip())

    return documents


# ========================= DATASET MANAGEMENT =========================

def _get_dataset(client: Client, dataset_name: str) -> Any:

    datasets = list(client.list_datasets(dataset_name=dataset_name))

    if datasets:
        print(f"[evaluation] Reusing dataset: {dataset_name}")
        return datasets[0]

    if not TEST_CASES:
        raise ValueError("Add test cases to TEST_CASES first.")

    for index, case in enumerate(TEST_CASES, start=1):
        if not all(
            str(case.get(field, "")).strip()
            for field in ("id", "question", "reference_answer")
        ):
            raise ValueError(
                f"Invalid TEST_CASES entry at position {index}."
            )

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="CampusQuery RAG evaluation dataset",
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {
                    "question": case["question"],
                    "test_case_id": case["id"],
                },
                "outputs": {
                    "answer": case["reference_answer"],
                },
            }
            for case in TEST_CASES
        ],
    )

    print(
        f"[evaluation] Created dataset '{dataset_name}' "
        f"with {len(TEST_CASES)} test cases."
    )

    return dataset


# ========================= MAIN EVALUATION =========================

def run_evaluation() -> Any:

    dataset_name = os.getenv(
        "LANGSMITH_EVAL_DATASET",
        "campusquery-evaluation",
    ).strip()

    experiment_name = os.getenv(
        "LANGSMITH_EVAL_EXPERIMENT",
        "campusquery-experiment",
    ).strip()

    application_model_name = _required_setting("GROQ_MODEL")
    judge_model_name = _required_setting("GROQ_JUDGE_MODEL")
    groq_api_key = _required_setting("GROQ_API_KEY")

    delay_seconds = float(
        os.getenv("EVALUATION_DELAY_SECONDS", "2")
    )
    max_retries = int(
        os.getenv("EVALUATION_MAX_RETRIES", "3")
    )

    if delay_seconds < 0 or max_retries < 0:
        raise ValueError("Delay and retries cannot be negative.")

    client = Client()

    # Get or create the dataset.
    dataset = _get_dataset(client, dataset_name)

    examples = list(client.list_examples(dataset_id=dataset.id))

    if not examples:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")

    for example in examples:
        if not (example.inputs or {}).get("question"):
            raise ValueError("A dataset example is missing its question.")

        if not (example.outputs or {}).get("answer"):
            raise ValueError("A dataset example is missing its reference answer.")

    # Preserve test case IDs, including for older datasets.
    case_ids_by_question = {}

    for example in examples:
        question = example.inputs["question"]
        case_id = (
            example.inputs.get("test_case_id")
            or str(example.id)
        )
        case_ids_by_question.setdefault(question, []).append(str(case_id))

    used_case_ids = {}

    # Initialize the judge LLM.
    judge_llm = ChatGroq(
        model=judge_model_name,
        groq_api_key=groq_api_key,
        temperature=0,
        max_retries=0,
    )

    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=judge_llm,
        feedback_key="correctness",
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        judge=judge_llm,
        feedback_key="groundedness",
    )

    store = get_vector_store(DEFAULT_CONFIG)

    scores = {
        "correctness": [],
        "groundedness": [],
    }

    case_results = {}

    # ========================= APPLICATION EXECUTION =========================

    def target(inputs: dict[str, Any]) -> dict[str, Any]:

        question = inputs["question"]

        case_id = inputs.get("test_case_id")

        if not case_id:
            ids = case_ids_by_question.get(question, [])
            index = used_case_ids.get(question, 0)

            case_id = ids[index] if index < len(ids) else "unknown"
            used_case_ids[question] = index + 1

        try:
            result = _run_with_retries(
                "RAG answer generation",
                lambda: answer_question(
                    question=question,
                    store=store,
                    config=DEFAULT_CONFIG,
                ),
                delay_seconds,
                max_retries,
            )

            chunks = result.get("retrieved_chunks") or []

            return {
                "test_case_id": str(case_id),
                "answer": str(result.get("answer", "")),
                "retrieved_context": _chunks_to_context(chunks),
            }

        except Exception as error:
            return {
                "test_case_id": str(case_id),
                "answer": "",
                "retrieved_context": [],
                "failure_reason": f"RAG pipeline failed: {error}",
            }

    # ========================= JUDGE EVALUATION =========================

    def make_evaluator(
        name: str,
        judge: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:

        def evaluator(
            inputs: dict[str, Any],
            outputs: dict[str, Any],
            reference_outputs: dict[str, Any] | None = None,
        ) -> dict[str, Any]:

            question = inputs.get("question", "")
            case_id = str(outputs.get("test_case_id", "unknown"))

            try:
                if outputs.get("failure_reason"):
                    raise RuntimeError(outputs["failure_reason"])

                judge_arguments = {
                    "inputs": {"question": question},
                    "outputs": {
                        "answer": outputs.get("answer", "")
                    },
                }

                if name == "correctness":
                    judge_arguments["reference_outputs"] = (
                        reference_outputs or {}
                    )
                else:
                    documents = outputs.get("retrieved_context", [])
                    if not documents:
                        raise ValueError(
                            "No retrieved document context was returned."
                        )
                    judge_arguments["context"] = {
                        "documents": documents
                    }

                result = _run_with_retries(
                    f"{name} evaluation",
                    lambda: judge(**judge_arguments),
                    delay_seconds,
                    max_retries,
                )

                passed = _get_score(result)

                explanation = str(
                    result.get("comment")
                    or result.get("reasoning")
                    or result.get("explanation")
                    or "No explanation returned."
                )

            except Exception as error:
                passed = False
                explanation = f"Evaluation failed: {error}"

            scores[name].append(passed)

            case_result = case_results.setdefault(
                case_id,
                {
                    "question": question,
                    "answer": outputs.get("answer", ""),
                    "failure_reason": outputs.get("failure_reason", ""),
                },
            )

            case_result[name] = passed
            case_result[f"{name}_explanation"] = explanation

            if explanation.startswith("Evaluation failed:"):
                case_result["failure_reason"] = (
                    case_result["failure_reason"] or explanation
                )

            if "correctness" in case_result and "groundedness" in case_result:

                print(f"\n{'=' * 55}")
                print(f"Test case: {case_id}")
                print(f"Question: {question}")
                print(f"Answer: {case_result['answer']}")

                print(
                    f"Correctness: "
                    f"{'PASS' if case_result['correctness'] else 'FAIL'}"
                )
                print(
                    f"Groundedness: "
                    f"{'PASS' if case_result['groundedness'] else 'FAIL'}"
                )
                print(
                    f"Correctness explanation: "
                    f"{case_result['correctness_explanation']}"
                )
                print(
                    f"Groundedness explanation: "
                    f"{case_result['groundedness_explanation']}"
                )

                print(
                    f"Failure reason: "
                    f"{case_result['failure_reason'] or 'None'}"
                )

                print(f"{'=' * 55}")

            return {
                "key": name,
                "score": passed,
                "comment": (
                    f"{'PASS' if passed else 'FAIL'}: {explanation}"
                ),
            }

        return evaluator

    # ========================= RUN EXPERIMENT =========================

    print(f"\nDataset: {dataset_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Application model: {application_model_name}")
    print(f"Judge model: {judge_model_name}")
    print(f"Test cases: {len(examples)}")

    results = evaluate(
        target,
        data=dataset.id,
        evaluators=[
            make_evaluator("correctness", correctness_judge),
            make_evaluator("groundedness", groundedness_judge),
        ],
        experiment_prefix=experiment_name,
        max_concurrency=1,
        client=client,
    )

    # ========================= FINAL REPORT =========================

    total = len(case_results)

    passed = sum(
        1
        for result in case_results.values()
        if result.get("correctness") is True
        and result.get("groundedness") is True
    )

    correctness_score = (
        sum(scores["correctness"]) / len(scores["correctness"])
        if scores["correctness"] else 0
    )

    groundedness_score = (
        sum(scores["groundedness"]) / len(scores["groundedness"])
        if scores["groundedness"] else 0
    )

    print("\n========== FINAL EVALUATION SUMMARY ==========")

    print(f"Total test cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Correctness pass rate: {correctness_score:.2%}")
    print(f"Groundedness pass rate: {groundedness_score:.2%}")

    print("\nFailed test cases:")

    for case_id, result in case_results.items():
        if result.get("correctness") is not True:
            print(f"\n{case_id}: {result['question']}")

            print(
                f"  Correctness: "
                f"{result.get('correctness_explanation', '')}"
            )

        if result.get("groundedness") is not True:
            print(f"\n{case_id}: {result['question']}")
            print(
                f"  Groundedness: "
                f"{result.get('groundedness_explanation', '')}"
            )

    experiment_url = getattr(results, "url", None)

    if experiment_url:
        print(f"\nLangSmith experiment: {experiment_url}")

    print("==============================================")

    return results


if __name__ == "__main__":
    run_evaluation()
