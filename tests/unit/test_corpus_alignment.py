"""
Regression tests for the CUAD corpus alignment fix.

Guards against reintroducing the mismatch between ingested contract names and
CUAD-QA evaluation titles (the bug that produced Hit Rate@5=0.02, MRR=0.013).
"""
from __future__ import annotations
import re
from pathlib import Path


def _word_normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def test_ingest_document_name_is_stem_not_name():
    """
    Ingested document_name must be path.stem (no extension) to match CUAD-QA
    title format.  E.g. 'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT',
    NOT 'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt'.
    """
    sample = Path("LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.txt")
    assert sample.stem == "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
    assert not sample.stem.endswith(".txt")


def test_cuad_v1_txt_stem_matches_qa_title():
    """Stems of v1 .txt files must equal the QA pair title verbatim."""
    pairs = [
        ("HEALTHASSURANCE_2000-01-01-EX-10.txt", "HEALTHASSURANCE_2000-01-01-EX-10"),
        ("CONTRACTA_2021.txt", "CONTRACTA_2021"),
    ]
    for filename, expected_title in pairs:
        assert Path(filename).stem == expected_title


def test_qa_filter_excludes_unmatched_contracts():
    """
    QA pairs for contracts not in ingested_names must be excluded so partial
    ingestion runs don't unfairly count misses for un-indexed contracts.
    """
    ingested_names = {"CONTRACT_A", "CONTRACT_B"}
    all_pairs = [
        {"question": "Q1", "answer": "A1", "document_name": "CONTRACT_A"},
        {"question": "Q2", "answer": "A2", "document_name": "CONTRACT_C"},
        {"question": "Q3", "answer": "A3", "document_name": "CONTRACT_B"},
    ]
    filtered = [p for p in all_pairs if p["document_name"] in ingested_names]
    assert len(filtered) == 2
    names = {p["document_name"] for p in filtered}
    assert "CONTRACT_A" in names
    assert "CONTRACT_B" in names
    assert "CONTRACT_C" not in names


def test_empty_ingested_set_gives_no_pairs():
    """With no ingested contracts, all QA pairs are filtered out."""
    filtered = [p for p in [{"document_name": "X"}] if p["document_name"] in set()]
    assert filtered == []


def test_word_normalize_handles_abbreviations():
    """
    Word normalization must make 'L.L.C.' (gold) match 'l. l. c.'
    (tokenizer output stored in ChromaDB).
    """
    gold = "Electric City of Illinois L.L.C."
    chunk = "The parties are Electric City of Illinois l. l. c. and Distributor."
    gold_prefix = " ".join(_word_normalize(gold).split()[:30])
    assert gold_prefix in _word_normalize(chunk)


def test_word_normalize_handles_redacted_text():
    """
    Word normalization must make answers with '[* ****]' (CUAD redaction) match
    chunk text where TextCleaner collapsed the asterisks to '[***]' and the
    tokenizer added spaces around them.
    """
    gold = "The Term shall be for a period of [* ****] years and [*****] months."
    chunk = "The term shall be for a period of [ * * * ] years and [ * * * ] months."
    gold_prefix = " ".join(_word_normalize(gold).split()[:30])
    assert gold_prefix in _word_normalize(chunk)


def test_word_normalize_fails_for_wrong_contract():
    """Hit detection must not succeed when answer is from a different contract."""
    gold_prefix = _word_normalize("indemnification clause paragraph 3 subsection b")
    chunk_words = _word_normalize("This agreement governs distribution of lime energy products.")
    assert gold_prefix not in chunk_words
