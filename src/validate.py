"""Check that every verse the LLM cites was really retrieved.

LLMs sometimes cite verses that don't exist, or that exist but were never shown
to them. This module catches both, in plain code (no LLM involved).

    result = validate(answer_text, allowed_refs={"BG 2.47", "BG 3.19"})
    result.ok            -> True if safe to show as-is
    result.invalid       -> refs cited that were NOT in the retrieved set
    clean = sanitize(answer_text, allowed_refs)   # strips bad refs, normalises good ones
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A bracket group that mentions BG, e.g. "[BG 2.47]", "[BG 2.47, 3.19]", "[BG 2.54-56]"
_BRACKET = re.compile(r"\[[^\[\]]*?\bBG\b[^\[\]]*\]", re.I)
# chapter.verse with optional range: 2.47   2.54-56
_REF = re.compile(r"(\d{1,2})\s*[.:]\s*(\d{1,3})(?:\s*[-\u2013]\s*(\d{1,3}))?")
# Uncited-style mentions in prose: "BG 4.7", "Gita 4.7", "verse 4.7"
_STRAY = re.compile(r"\b(?:BG|Gita|verses?)\s*(\d{1,2})\s*[.:]\s*(\d{1,3})\b", re.I)
# The model is told to say it couldn't find an answer; that is allowed to have no citations
_REFUSAL = re.compile(
    r"couldn['\u2019]?t find|could not find|not (?:directly )?(?:addressed|covered|mentioned)|"
    r"do(?:es)? not (?:directly )?(?:address|answer|cover)|no (?:relevant )?verses?|"
    r"(?:can['\u2019]?t|cannot|unable to) advise|consult (?:a|an|your) (?:qualified |licensed )?\w+",
    re.I,
)
# Strict check: the answer OPENS with a refusal. The prompt tells the model to start this way.
_OPENS_WITH_REFUSAL = re.compile(
    r"^\W*i (?:couldn['\u2019]?t|could not|can['\u2019]?t|cannot|am unable to|'m unable to) "
    r"(?:find|advise|help|answer)",
    re.I,
)


def _expand(m: re.Match) -> list[str]:
    ch, start = int(m.group(1)), int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    if end < start or end - start > 5:       # absurd ranges: treat as the first verse only
        end = start
    return [f"BG {ch}.{v}" for v in range(start, end + 1)]


def extract_citations(text: str) -> list[str]:
    """All refs cited inside [BG ...] brackets, in order of appearance, de-duplicated."""
    refs: list[str] = []
    for b in _BRACKET.finditer(text):
        for m in _REF.finditer(b.group(0)):
            for r in _expand(m):
                if r not in refs:
                    refs.append(r)
    return refs


def _stray_mentions(text: str) -> list[str]:
    outside = _BRACKET.sub(" ", text)
    return [f"BG {int(m.group(1))}.{int(m.group(2))}" for m in _STRAY.finditer(outside)]


@dataclass
class ValidationResult:
    cited: list[str] = field(default_factory=list)      # refs inside brackets
    valid: list[str] = field(default_factory=list)      # cited AND retrieved
    invalid: list[str] = field(default_factory=list)    # cited but NOT retrieved
    stray_invalid: list[str] = field(default_factory=list)  # prose mentions, not retrieved
    refused: bool = False                               # model said "couldn't find"
    has_citations: bool = False

    @property
    def ok(self) -> bool:
        if self.invalid or self.stray_invalid:
            return False
        return self.has_citations or self.refused

    def problems(self) -> list[str]:
        out = []
        if self.invalid:
            out.append(f"cited references that were not provided: {', '.join(self.invalid)}")
        if self.stray_invalid:
            out.append(f"mentioned verses that were not provided: {', '.join(self.stray_invalid)}")
        if not self.has_citations and not self.refused:
            out.append("gave no citations in the form [BG x.y]")
        return out


def validate(text: str, allowed_refs: set[str]) -> ValidationResult:
    cited = extract_citations(text)
    stray = [r for r in _stray_mentions(text) if r not in allowed_refs]
    return ValidationResult(
        cited=cited,
        valid=[r for r in cited if r in allowed_refs],
        invalid=[r for r in cited if r not in allowed_refs],
        stray_invalid=list(dict.fromkeys(stray)),
        refused=bool(_REFUSAL.search(text)),
        has_citations=bool(cited),
    )


def sanitize(text: str, allowed_refs: set[str]) -> str:
    """Last resort when retries fail: keep only verified citations.
    - bracket groups are rewritten as "[BG 2.47] [BG 3.19]" using only allowed refs
    - groups with nothing valid are removed
    - prose mentions of non-retrieved verses are replaced with a visible marker
    """
    def fix_bracket(b: re.Match) -> str:
        keep: list[str] = []
        for m in _REF.finditer(b.group(0)):
            for r in _expand(m):
                if r in allowed_refs and r not in keep:
                    keep.append(r)
        return " ".join(f"[{r}]" for r in keep)

    text = _BRACKET.sub(fix_bracket, text)

    def fix_stray(m: re.Match) -> str:
        ref = f"BG {int(m.group(1))}.{int(m.group(2))}"
        return m.group(0) if ref in allowed_refs else "[unverified reference removed]"

    text = _STRAY.sub(fix_stray, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s+([.,;:])", r"\1", text).strip()


def is_refusal(text: str) -> bool:
    """True if the answer opens with "I couldn't find..." / "I can't advise...".
    Citations inside such an answer are meaningless, so the pipeline strips them."""
    return bool(_OPENS_WITH_REFUSAL.match(text))


def strip_citations(text: str) -> str:
    """Remove every [BG ...] bracket and tidy the spacing/punctuation left behind."""
    text = _BRACKET.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([,;:])\s*(?=[.])", "", text)          # "word,." -> "word."
    return re.sub(r"\n{3,}", "\n\n", text).strip()
