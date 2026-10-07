from src.validate import extract_citations, is_refusal, sanitize, strip_citations, validate

ALLOWED = {"BG 2.47", "BG 3.19", "BG 2.54", "BG 2.55", "BG 2.56"}


def test_good_answer_passes():
    r = validate("Act without attachment [BG 2.47] and do your duty [BG 3.19].", ALLOWED)
    assert r.ok and r.valid == ["BG 2.47", "BG 3.19"] and not r.invalid


def test_faked_citation_is_caught():
    r = validate("The Gita says this clearly [BG 99.99].", ALLOWED)
    assert not r.ok and r.invalid == ["BG 99.99"]


def test_real_verse_that_was_not_retrieved_is_caught():
    # BG 4.7 exists in the Gita, but the model was never shown it
    r = validate("God descends when dharma declines [BG 4.7].", ALLOWED)
    assert not r.ok and r.invalid == ["BG 4.7"]


def test_comma_and_range_formats():
    assert extract_citations("see [BG 2.47, 3.19]") == ["BG 2.47", "BG 3.19"]
    assert extract_citations("see [BG 2.54-56]") == ["BG 2.54", "BG 2.55", "BG 2.56"]
    assert extract_citations("see [BG 2.47] [BG 2.47]") == ["BG 2.47"]


def test_stray_mention_outside_brackets_is_caught():
    r = validate("As BG 4.7 says, dharma declines. Also [BG 2.47].", ALLOWED)
    assert not r.ok and r.stray_invalid == ["BG 4.7"]


def test_stray_mention_of_retrieved_verse_is_fine():
    assert validate("In verse 2.47 we read this [BG 2.47].", ALLOWED).ok


def test_refusal_without_citations_is_ok():
    assert validate("I couldn't find this in the verses retrieved.", ALLOWED).ok


def test_uncited_answer_is_not_ok():
    r = validate("Karma yoga means acting selflessly.", ALLOWED)
    assert not r.ok and "no citations" in r.problems()[0]


def test_sanitize_removes_only_bad_refs():
    out = sanitize("Good [BG 2.47] but fake [BG 99.99] and mixed [BG 3.19, BG 98.1].", ALLOWED)
    assert "99.99" not in out and "98.1" not in out
    assert "[BG 2.47]" in out and "[BG 3.19]" in out
    assert validate(out, ALLOWED).ok


def test_sanitize_marks_stray_unverified_mentions():
    out = sanitize("As BG 4.7 says [BG 2.47].", ALLOWED)
    assert "unverified reference removed" in out and "BG 4.7" not in out


def test_is_refusal_only_when_answer_opens_with_one():
    assert is_refusal("I couldn't find this in the verses retrieved.")
    assert is_refusal("I can't advise on that. See a doctor.")
    assert is_refusal("  I cannot answer that.")
    assert not is_refusal("Act without attachment [BG 2.47]. I couldn't find more.")
    assert not is_refusal("The Gita says duty matters [BG 3.19].")


def test_strip_citations_leaves_clean_prose():
    out = strip_citations("I couldn't find this. The verses discuss duty [BG 3.2], devotion [BG 12.2]. [BG 1.1] [BG 2.2]")
    assert out == "I couldn't find this. The verses discuss duty, devotion."


def test_professional_referral_validates_without_citations():
    assert validate("I can't advise on medication. Please consult a qualified doctor.", ALLOWED).ok


def test_is_refusal_only_when_answer_opens_with_it():
    from src.validate import is_refusal
    assert is_refusal("I couldn't find this in the verses retrieved.")
    assert is_refusal("I can't advise on that. Please see a doctor.")
    assert not is_refusal("Act without attachment [BG 2.47]. I couldn't find more.")
    assert not is_refusal("The verses say this.")


def test_strip_citations_leaves_clean_prose():
    from src.validate import strip_citations
    out = strip_citations("I couldn't find this. They discuss duty [BG 3.2], devotion [BG 12.2]. [BG 1.1] [BG 2.2]")
    assert "BG" not in out and "[" not in out
    assert out == "I couldn't find this. They discuss duty, devotion."


def test_professional_referral_counts_as_valid_without_citations():
    assert validate("I can't advise on that. Please consult a qualified doctor.", ALLOWED).ok
