"""Stage 1a: download the raw Gita data and clean it into data/gita_verses.json.

Run:  python -m src.build_data
Each output record looks like:
  {"ref": "BG 2.47", "chapter": 2, "verse": 47,
   "sanskrit": "...", "transliteration": "...",
   "translation": "...", "translation_alt": "...", "word_meanings": "..."}
"""
from __future__ import annotations
import json
import re
import urllib.request

from src import config

FILES = ["verse.json", "translation.json"]


def download() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = config.RAW_DIR / name
        if dest.exists():
            print(f"  using cached {dest.name}")
            continue
        url = f"{config.RAW_BASE}/{name}"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)


def clean_sanskrit(text: str) -> str:
    """Drop the trailing verse marker (।।2.47।।) and tidy blank lines."""
    text = re.sub(r"।।\s*\d+\.\d+\s*।।", "", text)
    lines = [ln.strip() for ln in text.replace("\xa0", " ").split("\n")]
    return "\n".join(ln for ln in lines if ln)


def clean_plain(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s*\n\s*", " ", text)          # collapse newlines
    return re.sub(r"\s{2,}", " ", text).strip()


def clean_translation(text: str) -> str:
    """Some entries start with a verse marker like '2.47 ' or '।।2.47।।' - remove it."""
    text = re.sub(r"^\s*[।|]*\s*\d+\.\d+\s*[।|]*\s*", "", text.replace("\xa0", " "))
    text = re.sub(r"^\s*Revised:\s*", "", text)   # editorial artifact in a few entries
    return clean_plain(text)


_PLACEHOLDER = re.compile(r"^\s*(no changes( needed)?|n/?a|same as .*|see .*)\.?\s*$", re.I)


def is_placeholder(text: str) -> bool:
    """Some entries in the source data are editorial notes, not translations."""
    return len(text.strip()) < 20 or bool(_PLACEHOLDER.match(text))


def build() -> list[dict]:
    verses = json.loads((config.RAW_DIR / "verse.json").read_text(encoding="utf-8"))
    translations = json.loads((config.RAW_DIR / "translation.json").read_text(encoding="utf-8"))

    by_verse: dict[int, dict[int, str]] = {}
    for t in translations:
        if t["lang"] != "english":
            continue
        by_verse.setdefault(t["verse_id"], {})[t["author_id"]] = t["description"]

    out = []
    for v in sorted(verses, key=lambda x: (x["chapter_number"], x["verse_number"])):
        tr = by_verse.get(v["id"], {})
        primary = clean_translation(tr.get(config.TRANSLATION_ID, ""))
        alt = clean_translation(tr.get(config.TRANSLATION_ALT_ID, ""))
        if is_placeholder(primary) and not is_placeholder(alt):
            primary = alt          # fall back so users never see "No changes needed."
        out.append({
            "ref": f"BG {v['chapter_number']}.{v['verse_number']}",
            "chapter": v["chapter_number"],
            "verse": v["verse_number"],
            "sanskrit": clean_sanskrit(v["text"]),
            "transliteration": "\n".join(
                ln.strip() for ln in v["transliteration"].split("\n") if ln.strip()
            ),
            "translation": primary,
            "translation_alt": alt,
            "word_meanings": clean_plain(v.get("word_meanings", "")),
        })
    return out


def validate(records: list[dict]) -> None:
    """Cheap sanity checks so bad data never reaches the index."""
    assert len(records) == 701, f"expected 701 verses, got {len(records)}"
    refs = {r["ref"] for r in records}
    assert len(refs) == len(records), "duplicate refs"
    empties = [r["ref"] for r in records if not r["translation"] or not r["sanskrit"]]
    assert not empties, f"missing text for: {empties[:10]}"
    junk = [r["ref"] for r in records if is_placeholder(r["translation"])]
    assert not junk, f"placeholder/too-short translations: {junk}"
    per_chapter = {}
    for r in records:
        per_chapter[r["chapter"]] = per_chapter.get(r["chapter"], 0) + 1
    assert sorted(per_chapter) == list(range(1, 19)), "chapters should be 1..18"
    print("  validation passed:", len(records), "verses,", len(per_chapter), "chapters")


if __name__ == "__main__":
    print("Downloading raw data ...")
    download()
    print("Cleaning ...")
    records = build()
    validate(records)
    config.VERSES_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {config.VERSES_PATH}")
    for ref in ("BG 2.47", "BG 18.66"):
        r = next(x for x in records if x["ref"] == ref)
        print(f"\n--- {ref} ---\n{r['sanskrit']}\n{r['transliteration']}\n{r['translation']}")
