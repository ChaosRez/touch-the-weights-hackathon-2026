"""Phase 1 cold-run gate reporting."""

from examples.cartridge_loop import _summary


def _record(
    *, answer: str, submitted: bool, value_correct: float = 1.0, explicit: bool = True
) -> dict:
    record = {
        "answer": answer,
        "submitted": submitted,
        "metrics": {"value_correct": value_correct, "tool_calls": 1.0},
        "tool_executions": [{"name": "get_account"}],
    }
    if explicit:
        record["answered"] = bool(answer.strip())
    return record


def test_final_text_counts_as_an_answer_without_hiding_typed_submission_rate() -> None:
    records = [_record(answer="42", submitted=index < 7) for index in range(10)]

    summary = _summary(records, n=10)

    assert summary["first_ten_answer_rate"] == 1.0
    assert summary["first_ten_submission_rate"] == 0.7
    assert summary["tool_result_and_answer_episode"] is True


def test_empty_or_error_records_do_not_count_as_answers() -> None:
    records = [_record(answer="", submitted=False, value_correct=0.0), {"error": "boom"}]

    summary = _summary(records, n=2)

    assert summary["answered"] == 0
    assert summary["errors"] == 1
    assert summary["first_ten_answer_rate"] == 0.0


def test_pre_field_record_derives_answered_from_effective_answer() -> None:
    summary = _summary([_record(answer="42", submitted=False, explicit=False)], n=1)

    assert summary["answered"] == 1
    assert summary["first_ten_answer_rate"] == 1.0
