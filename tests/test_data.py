"""Unit tests for the TruthfulQA ROUGE grading oracle (no model, no network)."""

from data import grade_truthfulqa


def test_grade_correct_answer_labeled_0():
    label = grade_truthfulqa(
        "Paris is the capital of France.",
        correct_answers=["Paris is the capital of France.", "Paris."],
        incorrect_answers=["London is the capital of France."],
    )
    assert label == 0


def test_grade_incorrect_answer_labeled_1():
    label = grade_truthfulqa(
        "London is the capital of France.",
        correct_answers=["Paris is the capital of France."],
        incorrect_answers=["London is the capital of France.", "Berlin."],
    )
    assert label == 1


def test_grade_no_incorrect_references():
    label = grade_truthfulqa("anything", correct_answers=["anything"], incorrect_answers=[])
    assert label == 0


def test_grade_tie_favors_correct():
    label = grade_truthfulqa(
        "same text", correct_answers=["same text"], incorrect_answers=["same text"]
    )
    assert label == 0
