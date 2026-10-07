"""A deliberately simple pre-check that runs before the LLM.

If a message looks like a self-harm crisis, we return a fixed, compassionate reply
instead of asking a small local model to improvise. This is a safety net, not a
classifier: it will miss some phrasings and may over-trigger on some. Extend the
patterns as you see real questions (and review them with a mental-health professional
before a public launch).
"""
from __future__ import annotations

import re

from src import config

_CRISIS = re.compile(
    r"\b(?:kill(?:ing)? myself|end(?:ing)? my (?:own )?life|take my (?:own )?life|"
    r"suicid\w*|want(?:ed)? to die|wish i (?:was|were) dead|better off dead|"
    r"hurt(?:ing)? myself|self[- ]?harm\w*|no reason to live|don['\u2019]?t want to (?:live|be alive))\b",
    re.I,
)


def is_crisis(text: str) -> bool:
    return bool(_CRISIS.search(text))


def crisis_reply() -> str:
    msg = (
        "I'm really sorry you're going through something this painful. You deserve "
        "support from a real person right now, and a book or an app can't replace that.\n\n"
        "If you might act on these thoughts, or you're in immediate danger, please contact your "
        "local emergency number now, or reach out to a crisis helpline in your country. "
        "If you can, tell someone you trust how you're feeling today: a friend, a family member, "
        "or a doctor.\n"
    )
    if config.CRISIS_RESOURCES:
        msg += f"\n{config.CRISIS_RESOURCES}\n"
    msg += "\nYou don't have to carry this alone."
    return msg


# ----------------------------------------------------------------------------
# Personal medical / financial decisions -> fixed referral, LLM not called.
# A 7B model chose the wrong rule on this in evaluation (it said "couldn't find"
# instead of pointing to a professional), so we decide in code. Deliberately narrow:
# it needs BOTH a first-person decision cue AND a medical/financial keyword, so
# "What does the Gita say about medicine?" and "Should I invest in my relationships?"
# still go to the normal pipeline.
# ----------------------------------------------------------------------------
_PERSONAL_CUE = re.compile(
    r"\b(?:should i|can i|may i|do i need to|am i supposed to|is it (?:safe|ok|okay|fine|wise) (?:for me )?to|"
    r"i (?:want|plan|am planning|am thinking) (?:of |about )?to)\b", re.I)
_MEDICAL = re.compile(
    r"\b(?:medications?|medicines?|pills?|dosage|prescri\w+|antidepress\w+|insulin|chemotherapy|"
    r"surgery|vaccin\w+|diagnos\w+)\b", re.I)
_FINANCIAL = re.compile(
    r"\b(?:crypto\w*|bitcoin|stocks?|stock market|mutual funds?|mortgage|loans?|forex|day trading|"
    r"invest\w* (?:my|all|in (?:stocks?|crypto\w*|shares|funds?|property|real estate)))\b", re.I)

_REFERRALS = {
    "medical": "I can't advise on that. Decisions about medication or treatment should be made with a "
               "qualified doctor or pharmacist who knows your situation. The Gita can't guide that.",
    "financial": "I can't advise on that. Decisions about investing or your savings should be made with a "
                 "qualified financial advisor who knows your situation. The Gita can't guide that.",
}


def professional_referral(text: str):
    """Return a fixed referral message for personal medical/financial decisions, else None."""
    if not _PERSONAL_CUE.search(text):
        return None
    if _MEDICAL.search(text):
        return _REFERRALS["medical"]
    if _FINANCIAL.search(text):
        return _REFERRALS["financial"]
    return None


# ----------------------------------------------------------------------------
# Ranking religions / endorsing political parties -> fixed, neutral reply (LLM not called).
# A bare "I couldn't find this in the verses" reads as if an answer exists elsewhere;
# this says plainly that the app doesn't take sides. Narrow on purpose: "Which path is
# best, karma yoga or bhakti?" and "What does the Gita say about other religions?" are
# legitimate and still reach the normal pipeline.
# ----------------------------------------------------------------------------
_RELIGIONS = r"(?:hinduism|christianity|islam|buddhism|sikhism|judaism|jainism)"
_RELIGION_RANK = re.compile(
    r"\b(?:which|what)\s+(?:religion|faith|scripture|holy book)\s+is\s+(?:the\s+)?"
    r"(?:best|true|right|better|superior|greatest|real|correct)\b"
    rf"|\b{_RELIGIONS}\b[^.?!]*\b(?:better|superior|inferior|worse|truer|best|vs\.?|versus)\b"
    rf"|\b(?:better|superior|inferior|truer|best)\b[^.?!]*\b{_RELIGIONS}\b", re.I)
_PARTIES = r"(?:bjp|congress party|democrats?|republicans?|aap|trump|modi)"
_POLITICS = re.compile(
    r"\b(?:which|what)\s+(?:political\s+)?(?:party|politician|candidate)\b"
    r"|\bwho\s+(?:should|shall|do)\s+i\s+vote\b"
    rf"|\b{_PARTIES}\b[^.?!]*\b(?:support|endorse|approve|better|best|worse)\b"
    rf"|\b(?:support|endorse|approve|better|best|worse)\b[^.?!]*\b{_PARTIES}\b", re.I)

_POINTER = " I can explain what the Bhagavad Gita's verses say, so try asking about a specific teaching."
_DECLINES = {
    "religion": "I can't rank or compare religions." + _POINTER,
    "politics": "I can't take political positions or endorse parties." + _POINTER,
}


def neutral_decline(text: str):
    """Fixed neutral reply for 'which religion is best' / 'which party does the Gita back', else None."""
    if _RELIGION_RANK.search(text):
        return _DECLINES["religion"]
    if _POLITICS.search(text):
        return _DECLINES["politics"]
    return None
