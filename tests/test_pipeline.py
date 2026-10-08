"""Pipeline tests with a mock LLM and fake retrieval: no Ollama or Chroma needed."""
from src.pipeline import answer_question

VERSES = [
    {"ref": "BG 2.47", "chapter": 2, "verse": 47, "sanskrit": "s", "transliteration": "t",
     "translation": "You have the right to work, but not to its fruits."},
    {"ref": "BG 3.19", "chapter": 3, "verse": 19, "sanskrit": "s", "transliteration": "t",
     "translation": "Do your duty without attachment."},
]


def fake_retrieve(q, k=None):
    return VERSES


class ScriptedLLM:
    """Returns pre-written answers in order and records the messages it was sent."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.answers.pop(0)


def run(llm, q="What is karma yoga?"):
    def gita_related_llm(messages):
        if messages[0]["content"].startswith("Classify whether the user's question"):
            return "RELATED"
        return llm(messages)

    return answer_question(q, llm=gita_related_llm, retrieve_fn=fake_retrieve)


def test_good_answer_one_call():
    llm = ScriptedLLM("Act without craving results [BG 2.47] and do your duty [BG 3.19].")
    res = run(llm)
    assert res.attempts == 1 and res.flags == [] and not res.warning
    assert [v["ref"] for v in res.cited_verses] == ["BG 2.47", "BG 3.19"]


def test_prompt_contains_verses_and_question():
    llm = ScriptedLLM("Fine [BG 2.47].")
    run(llm)
    user_msg = llm.calls[0][1]["content"]
    assert "[BG 2.47]" in user_msg and "[BG 3.19]" in user_msg and "karma yoga" in user_msg
    assert llm.calls[0][0]["role"] == "system"


def test_prompt_includes_alternate_translations_and_word_meanings():
    verse = {
        **VERSES[0],
        "translation": "The son of Subhadra blew his conch.",
        "translation_alt": "Abhimanyu, the son of Subhadra and Arjuna, blew his conch.",
        "word_meanings": "saubhadraḥ—Abhimanyu, the son of Subhadra",
    }
    llm = ScriptedLLM("Abhimanyu is named [BG 2.47].")
    answer_question(
        "Does Krishna talk about Abhimanyu?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [verse],
    )
    prompt = llm.calls[0][1]["content"]
    assert "Alternate translation: Abhimanyu" in prompt
    assert "Word meanings: saubhadraḥ—Abhimanyu" in prompt


def test_fake_citation_triggers_retry_then_passes():
    llm = ScriptedLLM(
        "The Gita says so [BG 99.99].",                 # bad
        "Act without craving results [BG 2.47].",       # corrected
    )
    res = run(llm)
    assert res.attempts == 2 and "retried" in res.flags and "sanitized" not in res.flags
    assert [v["ref"] for v in res.cited_verses] == ["BG 2.47"]
    retry_prompt = llm.calls[1][-1]["content"]          # corrective feedback was sent
    assert "BG 99.99" in retry_prompt and "BG 2.47" in retry_prompt


def test_persistent_fake_citation_is_sanitized_with_warning():
    llm = ScriptedLLM("Claim one [BG 99.99].", "Claim again [BG 99.99] but also [BG 3.19].")
    res = run(llm)
    assert res.attempts == 2 and "sanitized" in res.flags and res.warning
    assert "99.99" not in res.text and "[BG 3.19]" in res.text
    assert [v["ref"] for v in res.cited_verses] == ["BG 3.19"]


def test_all_citations_fake_ends_flagged_uncited():
    llm = ScriptedLLM("Only fake [BG 99.99].", "Still fake [BG 98.98].")
    res = run(llm)
    assert "sanitized" in res.flags and "uncited" in res.flags and res.cited_verses == []


def test_honest_refusal_is_accepted_without_retry():
    llm = ScriptedLLM("I couldn't find this in the verses retrieved. They cover duty.")
    res = run(llm)
    assert res.attempts == 1 and res.flags == ["refused"] and not res.warning
    assert res.cited_verses == []


def test_refusal_returns_hybrid_results_matching_names_in_question():
    abhimanyu = {
        **VERSES[0],
        "translation_alt": "The son of Subhadra (Abhimanyu) was among the warriors.",
        "word_meanings": "saubhadraḥ—Abhimanyu, the son of Subhadra",
    }
    yama = {
        **VERSES[1],
        "ref": "BG 10.29",
        "translation_alt": "I am Yama among the governors.",
        "word_meanings": "yamaḥ—the celestial god of death",
    }

    for question, verse in (
        ("Does Krishna talk about Abhimanyu?", abhimanyu),
        ("Does Krishna talk about Shani and Yama?", yama),
    ):
        res = answer_question(
            question,
            llm=ScriptedLLM(
                "I couldn't find this in the verses retrieved.",
                f"The source identifies the name in the verse [{verse['ref']}].",
            ),
            retrieve_fn=lambda q, k=None, verse=verse: [verse],
        )
        assert res.attempts == 2
        assert res.flags == ["retried"]
        assert res.cited_verses == [verse]
        assert f"[{verse['ref']}]" in res.text


def test_persistent_refusal_returns_matching_retrieved_verses():
    verse = {
        **VERSES[0],
        "translation_alt": "I am Yama among the governors.",
        "word_meanings": "yamaḥ—the celestial god of death",
    }
    refusal = "I couldn't find this in the verses retrieved."
    res = answer_question(
        "Does Krishna talk about Shani and Yama?",
        llm=ScriptedLLM(refusal, refusal),
        retrieve_fn=lambda q, k=None: [verse],
    )

    assert res.attempts == 2
    assert res.flags == ["retried", "retrieval_fallback"]
    assert res.cited_verses == [verse]
    assert "may be relevant" in res.text
    assert res.warning


def test_refusal_with_spray_of_citations_strips_them_and_shows_no_cards():
    # real qwen behaviour from the evaluation: refuse, then list unrelated retrieved verses
    llm = ScriptedLLM("I couldn't find this in the verses retrieved. They cover duty. [BG 2.47] [BG 3.19]",
                      "I couldn't find this in the verses retrieved. They cover duty.")
    res = run(llm, "Write a Python function to sort a list.")
    assert res.attempts == 2 and res.flags == ["retried", "refused"]
    assert "[BG" not in res.text and res.cited_verses == []
    assert res.text.startswith("I couldn't find this")


def test_refusal_with_fabricated_citation_does_not_trigger_retry():
    # real llama3.2 behaviour: refusal followed by invented references
    llm = ScriptedLLM("I couldn't find any reference to that. [BG 99.99] [BG 98.98]")
    res = run(llm, "quote verse 19.5 about getting rich")
    assert res.attempts == 1 and "retried" not in res.flags and "99.99" not in res.text


def test_professional_referral_is_accepted_without_citations_or_retry():
    llm = ScriptedLLM("I can't advise on that. Please consult a qualified lawyer about your case.")
    res = run(llm, "My landlord is suing me, what do I do?")
    assert res.attempts == 1 and res.flags == ["refused"] and res.cited_verses == []
    assert not res.warning and "lawyer" in res.text


def test_retry_that_turns_into_refusal_is_accepted():
    llm = ScriptedLLM("Bad [BG 99.99].", "I couldn't find this in the verses retrieved.")
    res = run(llm)
    assert res.attempts == 2 and res.flags == ["retried", "refused"] and res.cited_verses == []


def test_prompt_demands_short_uncited_refusals_and_professional_referral():
    from src.generate import SYSTEM_PROMPT
    assert "begin exactly with" in SYSTEM_PROMPT and "Do not cite any verse" in SYSTEM_PROMPT
    assert "qualified professional" in SYSTEM_PROMPT
    assert "For \"how many\" questions" in SYSTEM_PROMPT


def test_crisis_never_calls_llm():
    llm = ScriptedLLM()   # would raise IndexError if called
    res = run(llm, "I want to end my life")
    assert res.flags == ["crisis"] and llm.calls == []


def test_no_retrieval_unrelated_question_skips_web_search():
    llm = ScriptedLLM("DECLINE")
    searches = []

    res = answer_question(
        "What is 2+2?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [],
        web_search_fn=lambda q: searches.append(q) or WEB_RESULTS,
    )
    assert res.flags == ["unrelated"]
    assert "focused on Bhagavad Gita" in res.text
    assert searches == []
    assert len(llm.calls) == 1


WEB_RESULTS = [{
    "title": "A relevant reference",
    "url": "https://example.com/reference",
    "content": "A factual detail supported by this page.",
}]


def test_empty_retrieval_uses_web_search_and_cites_its_source():
    llm = ScriptedLLM("SEARCH_WEB", "The answer is supported by this page [1].")
    searches = []

    def search(q):
        searches.append(q)
        return WEB_RESULTS

    res = answer_question(
        "What do historians say about the Bhagavad Gita's composition?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [],
        web_search_fn=search,
    )

    assert searches == ["What do historians say about the Bhagavad Gita's composition?"]
    assert res.flags == ["web_fallback"]
    assert res.web_sources == WEB_RESULTS
    assert "[1]" in res.text and res.warning == ""
    assert "A factual detail supported by this page." in llm.calls[1][1]["content"]


def test_unrelated_vector_results_do_not_use_web_search():
    llm = ScriptedLLM("I couldn't find this in the verses retrieved.")
    irrelevant_verses = [{
        **VERSES[0],
        "found_by": ["vector#1"],
        "vector_distance": 0.51,
    }]
    res = answer_question(
        "How do I make pasta carbonara?",
        llm=llm,
        retrieve_fn=lambda q, k=None: irrelevant_verses,
        web_search_fn=lambda q: WEB_RESULTS,
    )

    assert res.flags == ["refused"]
    assert res.web_sources == []
    assert res.cited_verses == []
    assert len(llm.calls) == 1


def test_close_vector_results_skip_relevance_check_and_answer_from_verses():
    llm = ScriptedLLM("Act without attachment [BG 2.47].")
    close_but_unrelated_verse = {
        **VERSES[0],
        "found_by": ["vector#1"],
        "vector_distance": 0.435,
    }
    searches = []
    res = answer_question(
        "What is 2+2?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [close_but_unrelated_verse],
        web_search_fn=lambda q: searches.append(q) or WEB_RESULTS,
    )

    assert res.flags == []
    assert searches == []
    assert len(llm.calls) == 1
    assert res.cited_verses == [close_but_unrelated_verse]


def test_relevant_verses_do_not_use_web_search():
    llm = ScriptedLLM("Act without attachment [BG 2.47].")
    verse = {**VERSES[0], "found_by": ["vector#1"], "vector_distance": 0.31}
    searches = []
    res = answer_question(
        "What is karma yoga?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [verse],
        web_search_fn=lambda q: searches.append(q) or WEB_RESULTS,
    )

    assert searches == []
    assert res.web_sources == []
    assert res.cited_verses == [verse]


def test_direct_verse_lookup_does_not_use_web_search():
    llm = ScriptedLLM("Act without attachment [BG 2.47].")
    verse = {**VERSES[0], "found_by": ["reference"], "vector_distance": 0.99}
    searches = []
    answer_question(
        "Explain BG 2.47",
        llm=llm,
        retrieve_fn=lambda q, k=None: [verse],
        web_search_fn=lambda q: searches.append(q) or WEB_RESULTS,
    )
    assert searches == []


def test_invalid_web_source_citations_are_removed():
    llm = ScriptedLLM("SEARCH_WEB", "Supported claim [1]. Unsupported claim [8].")
    res = answer_question(
        "What do historians say about the Bhagavad Gita's composition?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [],
        web_search_fn=lambda q: WEB_RESULTS,
    )
    assert res.text == "Supported claim [1]. Unsupported claim ."
    assert res.warning == "An invalid web-source citation was removed."


def test_web_search_failure_is_reported_after_relevance_check():
    from src.web_search import WebSearchError

    def failed_search(_):
        raise WebSearchError("Tavily is unavailable.")

    llm = ScriptedLLM("SEARCH_WEB")
    res = answer_question(
        "What do historians say about the Bhagavad Gita's composition?",
        llm=llm,
        retrieve_fn=lambda q, k=None: [],
        web_search_fn=failed_search,
    )
    assert res.flags == ["no_retrieval"]
    assert res.warning == "Tavily is unavailable."
    assert len(llm.calls) == 1


def test_invalid_agent_tool_choice_fails_closed():
    llm = ScriptedLLM("Maybe Gita-related, maybe not.")
    searches = []
    res = answer_question(
        "Unclear question",
        llm=llm,
        retrieve_fn=lambda q, k=None: [],
        web_search_fn=lambda q: searches.append(q) or WEB_RESULTS,
    )

    assert res.flags == ["relevance_uncertain"]
    assert "no search tool was run" in res.warning
    assert searches == []


def test_bounded_agent_can_rewrite_query_for_one_verse_search():
    llm = ScriptedLLM("SEARCH_VERSES: son of Subhadra Abhimanyu Saubhadra", "Abhimanyu appears in [BG 1.6].")
    verse = {
        **VERSES[0],
        "ref": "BG 1.6",
        "translation_alt": "The son of Subhadra (Abhimanyu) was among the warriors.",
    }
    calls = []

    def retrieve(query, k=None):
        calls.append(query)
        return [] if len(calls) == 1 else [verse]

    res = answer_question(
        "Does Krishna talk about Abhimanyu?",
        llm=llm,
        retrieve_fn=retrieve,
    )

    assert calls == ["Does Krishna talk about Abhimanyu?", "son of Subhadra Abhimanyu Saubhadra"]
    assert res.flags == ["agent_verse_search"]
    assert res.cited_verses == [verse]


def test_refusal_citing_real_verses_gets_one_chance_to_decide():
    # evaluation c19: "couldn't find this [BG 5.2] [BG 5.1]" although those verses answered it
    llm = ScriptedLLM("I couldn't find this in the verses retrieved. [BG 2.47] [BG 3.19]",
                      "Right action is the better path [BG 3.19].")
    res = run(llm, "Is renunciation better than action?")
    assert res.attempts == 2 and res.flags == ["retried"] and not res.warning
    assert [v["ref"] for v in res.cited_verses] == ["BG 3.19"]
    assert "couldn't find this" in llm.calls[1][-1]["content"]      # the nudge was sent


def test_refusal_spray_that_is_really_a_refusal_ends_clean():
    llm = ScriptedLLM("I couldn't find this in the verses retrieved. [BG 2.47] [BG 3.19]",
                      "I couldn't find this in the verses retrieved.")
    res = run(llm, "Write a Python function to sort a list.")
    assert res.attempts == 2 and res.flags == ["retried", "refused"]
    assert "BG" not in res.text and res.cited_verses == []


def test_persistent_refusal_spray_is_stripped_after_the_retry_budget():
    spray = "I couldn't find this in the verses retrieved. [BG 2.47]"
    res = run(ScriptedLLM(spray, spray))
    assert res.attempts == 2 and res.flags == ["retried", "refused"] and "BG" not in res.text


def test_refusal_after_a_failed_first_attempt():
    llm = ScriptedLLM("Claim [BG 99.99].", "I couldn't find this in the verses retrieved.")
    res = run(llm)
    assert res.attempts == 2 and res.flags == ["retried", "refused"] and res.cited_verses == []


def test_personal_medical_question_gets_fixed_referral_without_llm():
    llm = ScriptedLLM()
    res = run(llm, "Should I stop taking my medication?")
    assert res.flags == ["referred"] and llm.calls == [] and "qualified doctor" in res.text
    assert res.cited_verses == [] and res.retrieved == []


def test_personal_financial_question_gets_fixed_referral_without_llm():
    llm = ScriptedLLM()
    res = run(llm, "Should I put all my savings into cryptocurrency?")
    assert res.flags == ["referred"] and llm.calls == [] and "financial advisor" in res.text


def test_general_gita_questions_about_medicine_or_investing_still_reach_the_llm():
    for q in ("What does the Gita say about medicine?", "Should I invest in my relationships?"):
        llm = ScriptedLLM("Act with care [BG 2.47].")
        res = run(llm, q)
        assert res.attempts == 1 and "referred" not in res.flags, q


def test_ranking_religions_or_parties_gets_fixed_neutral_reply_without_llm():
    for q, expect in (("Which religion is the best, Hinduism or Christianity?", "religions"),
                      ("Which political party does the Gita support?", "political")):
        llm = ScriptedLLM()
        res = run(llm, q)
        assert res.flags == ["declined"] and llm.calls == [] and expect in res.text and res.cited_verses == []


def test_comparing_yoga_paths_still_reaches_the_llm():
    llm = ScriptedLLM("Both lead to the goal [BG 2.47].")
    res = run(llm, "Which path is best, karma yoga or bhakti yoga?")
    assert res.attempts == 1 and "declined" not in res.flags
