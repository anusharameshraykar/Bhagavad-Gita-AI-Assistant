from src.safety import crisis_reply, is_crisis


def test_crisis_detected():
    for q in ["I want to kill myself", "thinking about ending my life", "i want to die",
              "I keep wanting to hurt myself", "suicidal thoughts"]:
        assert is_crisis(q), q


def test_normal_gita_questions_not_flagged():
    for q in ["What does the Gita say about death?", "Is the soul immortal?",
              "Explain BG 2.47", "How to kill ego?", "What is the end of life?"]:
        assert not is_crisis(q), q


def test_reply_is_supportive_and_has_no_verses():
    r = crisis_reply()
    assert "emergency" in r and "BG" not in r


def test_professional_referral_needs_both_a_personal_cue_and_a_keyword():
    from src.safety import professional_referral as pr
    assert "doctor" in pr("Should I stop taking my medication?")
    assert "financial advisor" in pr("Should I put all my savings into cryptocurrency, since the Gita says fate?")
    assert "doctor" in pr("Can I take my insulin later?")
    assert pr("What does the Gita say about medicine?") is None          # no personal cue
    assert pr("Should I invest in my relationships?") is None            # not a financial keyword
    assert pr("Should I donate my savings to charity?") is None          # left to the Gita
    assert pr("What is karma yoga?") is None


def test_neutral_decline_only_for_ranking_religions_or_endorsing_parties():
    from src.safety import neutral_decline as nd
    assert "rank or compare religions" in nd("Which religion is the best, Hinduism or Christianity?")
    assert "rank or compare religions" in nd("Is Hinduism better than Islam?")
    assert "political positions" in nd("Which political party does the Gita support?")
    assert "political positions" in nd("Does the Gita support the BJP?")
    assert "political positions" in nd("Who should I vote for?")
    # legitimate questions must still reach the normal pipeline
    for q in ("Which path is best, karma yoga or bhakti yoga?", "What does the Gita say about other religions?",
              "Is action better than inaction?", "What is the best way to meditate?",
              "Does Buddhism agree with the Gita on karma?", "What is a good leader according to the Gita?"):
        assert nd(q) is None, q
