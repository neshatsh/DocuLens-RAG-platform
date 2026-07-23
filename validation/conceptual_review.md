# DocuLens — Conceptual Soundness Review

**Scope:** Independent review of DocuLens's design choices, based on direct
inspection of the ingestion, retrieval, and generation code (not the README
description alone). This is a conceptual review — it identifies design risks
and reasons about failure modes, prior to any empirical testing.

**Reviewer note:** this document intentionally separates *design risk* from
*measured performance*. A design choice can be reasonable and still perform
poorly, or be questionable and still perform well on a given benchmark — the
empirical eval (`faithfulness_eval.py`, `reproducibility_test.py`, etc.) is
where we find out which is which. This document is the "why might this be
wrong" pass; it does not claim any of these risks were observed in practice.

---

## 1. Chunking: sentence-boundary packing with token overlap

**What it does:** `SemanticChunker` splits on sentence boundaries via a regex
(`re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])"`), then greedily packs sentences into
token-bounded batches with a fixed-token overlap carried from the tail of the
previous chunk.

**Design risk 1 — the sentence splitter is not legal-text-aware.**
The regex assumes a sentence ends at `.!?` followed by whitespace and a capital
letter. Legal contracts routinely violate this:
- Numbered/lettered sub-clauses (`(a) ... ; (b) ...`) often don't end in
  terminal punctuation at all, or use semicolons as clause separators.
- Abbreviations and defined terms (`Corp.`, `Inc.`, `Sec. 4.2`, `U.S.C.`) will
  falsely trigger a sentence break.
- Section headers in ALL CAPS or numbered outlines (`4.2 GOVERNING LAW.`) can
  confuse the capital-letter lookahead.

The consequence isn't just cosmetic — a clause and its exception/proviso
("...except as set forth in Section 4.2...") can end up split across chunks if
the false-positive sentence break lands between them. Since the reranker and
LLM only see the chunk's own text, an exception could be silently dropped from
the answer even though it was present in the source document, without the
system being able to detect this happened.

**Design risk 2 — the overlap window is measured in tokens, not clauses.**
`overlap_tokens` is a fixed count carried from the end of the previous batch,
independent of clause structure. This helps with local context continuity but
doesn't guarantee a full clause number, defined term, or cross-reference
("as defined in Section 2.1") survives into the next chunk. A clause reference
split exactly at the token-overlap boundary could still lose its antecedent.

**What would mitigate this:** clause-aware splitting (respecting numbered/
lettered sub-item boundaries as hard stops) would likely reduce this failure
mode more than tuning `overlap_tokens`, since the risk is structural, not just
a window-size problem.

---

## 2. Embedding: general-purpose sentence embeddings for legal/financial text

**What it does:** `all-MiniLM-L6-v2` (384-dim, general-domain
sentence-transformer) embeds each chunk; cosine similarity via normalized
embeddings.

**Design risk — the model was not trained on legal or financial language.**
`all-MiniLM-L6-v2` is trained on general-domain sentence-similarity data. Legal
text has domain-specific properties this model has no particular reason to
capture well:
- **Negation sensitivity.** "Party A shall indemnify Party B" and "Party A
  shall *not* indemnify Party B" are near-identical in surface form and are
  not guaranteed to be well-separated in general-purpose embedding space —
  negation handling is a known weak point for sentence embeddings not
  fine-tuned to be negation-sensitive.
- **Defined-term dependence.** Legal meaning often hinges on capitalized
  defined terms ("the Agreement," "Confidential Information") whose scope is
  set elsewhere in the document. A general embedding model has no signal that
  "the Agreement" here refers to a specific 40-page document defined in a
  recital three pages earlier.
- **Numeric/date precision.** Cosine similarity over general embeddings
  doesn't reliably distinguish "30 days" from "90 days" as semantically
  different in a way that matters for the question being asked, even though
  they're critical for e.g. termination-notice clauses.

**What would mitigate this:** a legal-domain-tuned embedding model (or at
minimum, an empirical check specifically targeting negation and numeric
precision, see recommended tests below) would directly address this risk. This
is a reasonable and common choice for a first version — the risk is worth
flagging, not necessarily worth rearchitecting around before evidence of
impact.

---

## 3. Retrieval: two-stage bi-encoder → cross-encoder (top-20 → top-5)

**What it does:** Dense bi-encoder retrieval fetches top-20 by cosine
similarity, then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescoring narrows
to top-5. This is a standard, well-justified architecture — the code and
docstring both correctly explain the rationale (bi-encoder for recall/speed,
cross-encoder for precision).

**Design risk — the reranker was trained on web/QA passage ranking (MS
MARCO), not legal contract text.**
`ms-marco-MiniLM-L-6-v2` is trained on short web-search query/passage pairs.
It's a reasonable general-purpose reranker, but its notion of "relevance" was
learned from a very different distribution than "which clause among 20
candidate contract excerpts actually answers this specific legal question."
This is the same category of risk as the embedding model, one level later in
the pipeline.

**Design risk — fixed top-20/top-5 window regardless of corpus size or query
type.**
For document-scoped queries (asking about one specific contract), a fixed
top-20 pulled from the *entire* corpus is a materially harder retrieval
problem than the same top-20 pulled from just that contract's chunks — this is
exactly the distinction the project's own `--per-document` vs. cross-corpus
evaluation modes surface (measured Hit Rate@5 of 0.71 vs. 0.13 respectively).
The architecture doesn't adapt window size to whether a `document_filter` is
active, though the retriever's API does support filtering.

---

## 4. Generation: prompt-enforced citation, no verification layer

**What it does:** `PromptBuilder` constructs a system prompt instructing the
LLM to cite every claim as `[Source: <document_name>, Page <page_number>]`,
answer only from provided context, and flag contradictions — all via prompt
engineering. There is no code-level verification that the LLM actually
followed these instructions.

**Design risk — citation correctness is entirely trust-based.**
Nothing in `answer_generator.py` checks that:
- a cited `[Source: ..., Page ...]` actually corresponds to a chunk that was
  in the retrieved context (the LLM could cite a real document/page that
  wasn't retrieved, or hallucinate one that doesn't exist),
- the claim attributed to a citation is actually supported by that chunk's
  text (the LLM could cite the right chunk for the wrong reason), or
- the "if the context does not contain enough information, say so" instruction
  is honored under adversarial or ambiguous conditions.

This is a prompt-only faithfulness control — production-grade in the sense
that it's the standard first line of defense, but it has a known failure
mode class (instruction-following degrades under distractor context, long
context, or adversarial input) with no downstream check to catch it. This is
precisely what a faithfulness eval (ragas `faithfulness`/`context_precision`)
and a prompt-injection red-team are positioned to test empirically — this
review predicts a possible gap, not a confirmed one.

**Design risk — no confidence signal distinguishes "I found the answer" from
"I found something plausible-looking."**
The reranker's score is available (`rerank_score`) and surfaced in the answer
metadata, but the prompt doesn't instruct the LLM to condition its confidence
language on that score, and there's no threshold below which the system
declines to answer rather than answering off a weak match. A low-relevance
top chunk can still produce a confidently-worded answer.

---

## 5. What this review does *not* cover

This is a design-level review, not a security review or a full failure-mode
enumeration. Notably out of scope here (covered by other planned validation
work instead):
- VLM extraction quality on scanned/low-quality pages — no confidence signal
  currently surfaces if OCR-adjacent extraction silently fails or degrades.
- Reproducibility/determinism of LLM generation across repeated identical
  queries.
- Adversarial robustness (prompt injection via document content).

---

## Summary table

| Component | Design choice | Primary risk | Severity (design-level, unverified) |
|---|---|---|---|
| Chunking | Sentence-boundary regex splitter | Legal text structure (sub-clauses, defined terms) can break sentence detection | Medium |
| Chunking | Fixed token-count overlap | Clause/reference continuity not guaranteed across chunk boundary | Medium |
| Embedding | General-domain `all-MiniLM-L6-v2` | Negation, defined-term scope, numeric precision may not be well-separated | Medium |
| Retrieval | Fixed top-20 window, corpus-agnostic | Cross-corpus retrieval is a harder problem than document-scoped, window doesn't adapt | Low–Medium (mitigated by `document_filter` support) |
| Reranking | MS MARCO-trained cross-encoder | Domain mismatch with legal relevance judgments | Low–Medium |
| Generation | Prompt-only citation enforcement | No verification that citations are grounded or accurate | **High** — this is the most consequential gap, since it's the user-facing trust layer |
| Generation | No confidence gating | Low-relevance matches can still yield confident-sounding answers | Medium |

The generation-layer citation risk is flagged as highest severity because it's
the layer users directly rely on — a faithfulness gap here means a
professional-looking, well-cited answer could still misrepresent the source
document, which is the single most consequential failure mode for a legal/
financial document tool.
