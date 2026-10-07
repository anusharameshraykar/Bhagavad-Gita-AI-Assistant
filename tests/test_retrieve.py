from src.retrieve import parse_references, tokenize, normalize

VALID = {f"BG {c}.{v}" for c in range(1, 19) for v in range(1, 80)}


def test_reference_formats():
    assert parse_references("explain 2.47", VALID) == ["BG 2.47"]
    assert parse_references("BG 2:47 please", VALID) == ["BG 2.47"]
    assert parse_references("Chapter 2, verse 47", VALID) == ["BG 2.47"]
    assert parse_references("ch 18 v 66", VALID) == ["BG 18.66"]
    assert parse_references("verse 47 of chapter 2", VALID) == ["BG 2.47"]


def test_multiple_and_dedupe_and_order():
    assert parse_references("compare 3.19 and 2.47", VALID) == ["BG 3.19", "BG 2.47"]
    assert parse_references("2.47 ... again 2.47", VALID) == ["BG 2.47"]


def test_invalid_references_ignored():
    assert parse_references("chapter 19 verse 1", VALID) == []   # no chapter 19
    assert parse_references("what is karma yoga?", VALID) == []
    assert parse_references("I scored 100.5 points", VALID) == []


def test_normalize_strips_diacritics():
    assert normalize("śraddhā") == "sraddha"
    assert normalize("Dhṛitarāśhtra") == "dhritarashtra"


def test_hyphenated_compounds_indexed_split_and_joined():
    toks = tokenize("sthita-prajñasya")
    assert "sthita" in toks and "prajnasya" in toks and "sthitaprajnasya" in toks
