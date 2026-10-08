import json
import pytest

from eval.run_eval import (MockLLM, QUESTIONS_PATH, build_report, expected_hit, load_questions,
                           run_one, score_adversarial, summarize)
from src import config

QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def test_question_set_shape():
    assert len(QUESTIONS) == 36
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for q in QUESTIONS:
        assert q["type"] in ("concept", "reference", "adversarial")
        if q["type"] != "adversarial":
            assert q["expect"], q["id"]
        else:
            assert q.get("must_match") or q.get("must_not_match") or q.get("expect_flag"), q["id"]


def test_expected_verses_exist_in_dataset():
    if not config.VERSES_PATH.exists():
        pytest.skip("run `python -m src.build_data` first")
    refs = {v["ref"] for v in json.loads(config.VERSES_PATH.read_text(encoding="utf-8"))}
    for q in QUESTIONS:
        for e in q.get("expect", []):
            assert e in refs, f"{q['id']}: {e} is not a real verse in the dataset"


def test_dataset_has_no_placeholder_translations():
    if not config.VERSES_PATH.exists():
        pytest.skip("run `python -m src.build_data` first")
    for v in json.loads(config.VERSES_PATH.read_text(encoding="utf-8")):
        assert len(v["translation"]) >= 20 and "no changes" not in v["translation"].lower(), v["ref"]


def test_expected_hit_any_vs_all():
    assert expected_hit({"expect": ["BG 2.47", "BG 3.19"]}, ["BG 3.19"])
    assert not expected_hit({"expect": ["BG 2.47", "BG 3.19"], "expect_all": True}, ["BG 3.19"])
    assert expected_hit({"expect": ["BG 2.47", "BG 3.19"], "expect_all": True}, ["BG 3.19", "BG 2.47"])
    assert not expected_hit({"expect": []}, ["BG 2.47"])


def test_adversarial_checks():
    q = {"must_match": "doctor", "must_not_match": "you should stop"}
    assert all(score_adversarial(q, "Please ask your doctor.", []).values())
    assert not all(score_adversarial(q, "You should stop taking it.", []).values())
    assert not all(score_adversarial(q, "Ask a friend.", []).values())      # no professional mentioned
    assert score_adversarial({"expect_flag": "crisis"}, "x", ["crisis"]) == {"flag:crisis": True}


VERSES = [{"ref": "BG 2.47", "chapter": 2, "verse": 47, "sanskrit": "s", "transliteration": "t", "translation": "tr"}]


def test_run_one_and_summary_with_mock():
    llm = MockLLM()
    rows = [run_one(q, llm, VERSES, 6) for q in QUESTIONS if q["id"] in ("c01", "r01", "a01", "a06")]
    by = {r["id"]: r for r in rows}
    assert by["r01"]["retrieval_hit"] and by["r01"]["cited_expected"]
    assert by["c01"]["retrieval_hit"] and by["c01"]["cited_expected"]
    assert by["a06"]["checks"] == {"flag:crisis": True} and by["a06"]["attempts"] == 0
    s = summarize(rows)
    assert s["cited_expected"] == 1.0 and s["adversarial_auto_pass"] <= 1.0
    report = build_report({"mock": rows}, 6, [q for q in QUESTIONS if q["id"] in by])
    assert "## Summary" in report and "a06" in report


def test_load_questions_filters():
    assert len(load_questions(only="adversarial")) == 7
    assert len(load_questions(limit=3)) == 3
