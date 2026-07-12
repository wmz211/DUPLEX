# Adaptive Qwen Retrieval Design

## Objective

Improve factuality verification for long news texts without atomizing every sentence or forcing every sample through two search rounds. Keep the existing dual-path architecture, but move iterative retrieval from the Bocha semantic path to the Qwen verification path.

The loop changes only the evidence context available to Qwen. It does not change the existing `Real`/`Factuality` label definitions, confidence scoring rules, or calibrated fusion formula.

## Evidence from Existing Runs

The latest `dev50_rerun` Investigator output contains 179 evaluated samples. Treating `semantic_score >= 55` as a rough semantic-path preference for `Real`:

- 54 samples had no path conflict and were classified correctly.
- 115 samples had a path conflict but were classified correctly.
- 10 samples had a path conflict and were classified incorrectly.
- No incorrect sample occurred without a path conflict.

All 10 errors had high Qwen confidence. Nine were real texts that Qwen labeled `Factuality` with confidence between 82 and 92. One was a factual hallucination that Qwen labeled `Real` with confidence 90. Therefore:

- Path conflict alone is too broad as a loop trigger because it occurs in 69.8% of samples.
- Low Qwen confidence alone cannot identify the known errors.
- The loop should be triggered by the sufficiency and auditability of Qwen's evidence, not confidence alone.

## Architecture

### Bocha Semantic Path

The Bocha path becomes a fixed one-round path:

1. DeepSeek produces one search query and three key facts, preserving the current prompt behavior.
2. Bocha returns the top three results.
3. The system compares the key facts with each result's title and snippet using SentenceTransformer cosine similarity.
4. The path outputs semantic relevance, coverage, the query, key facts, and search-result metadata.

This path does not decide whether evidence supports or refutes the news. It does not rewrite its query and does not loop. Its score remains an independent semantic relevance prior for calibrated fusion.

### Qwen Verification Path

The Qwen path performs one or two rounds with `enable_search=True`:

1. Round one freely fact-checks the complete news text. Qwen decides which facts are important; there is no fixed number of claims and no complete atomic decomposition.
2. Qwen reports its verdict and confidence using the existing scoring definitions. It also reports the evidence used, sources when available, important unverified information, whether more search is required, and a focused follow-up query.
3. A deterministic evidence-sufficiency gate decides whether to run round two.
4. Round two receives the original text plus the complete round-one state. It searches only for missing information or checks claims made by the first round.
5. Round two consolidates the retained first-round evidence and new second-round evidence, rejects evidence it finds incorrect, and produces a final cumulative verdict and confidence.

Qwen runs at most twice per text.

### Evidence Sufficiency Gate

The gate does not call another model and does not determine factuality. It starts round two when any of the following holds:

- `need_more_search` is true.
- `unverified_points` is non-empty.
- Qwen's verdict is `Factuality`, but the response does not provide concrete evidence for the alleged error.
- The response lacks the required evidence summary.
- The structured output cannot be parsed reliably.

Confidence alone is not a loop trigger or a stopping condition.

## Qwen Data Contracts

### Round One

Round one retains free-form fact selection while returning a structured envelope:

```json
{
  "verdict": "Real | Factuality",
  "confidence": 88,
  "evidence_summary": "Evidence used for the first-round decision",
  "sources": ["Source name or URL when available"],
  "unverified_points": [
    "Important information not verified in this round"
  ],
  "need_more_search": true,
  "next_query": "Focused query for the most important evidence gap"
}
```

Qwen remains responsible for deciding which details matter. The output does not enumerate every atomic fact in a long article.

### Round Two Input

Round two receives:

- The original news text.
- The round-one verdict and confidence.
- The round-one evidence summary and sources.
- The round-one unverified points.
- The proposed follow-up query.

It is explicitly instructed not to restart a complete analysis of already verified material.

### Round Two Output

Round two returns a cumulative assessment:

```json
{
  "new_evidence": "Evidence newly found in round two",
  "retained_evidence": "Round-one evidence that remains valid",
  "rejected_evidence": "Round-one evidence corrected or discarded",
  "remaining_unverified_points": [],
  "final_verdict": "Real | Factuality",
  "final_confidence": 87,
  "final_evidence_summary": "Assessment based on all currently available information"
}
```

The second round does not receive an independent score that is averaged mechanically with round one. It reassesses the article using the accumulated evidence `E1 + E2`. This prevents an incorrect high-confidence first-round claim from blocking a valid correction.

## Scoring and Fusion

The existing confidence definitions remain unchanged. If round two finds no new evidence or important information remains unverified, Qwen still evaluates the article using all currently available information and returns `Real` or `Factuality` with confidence under the existing rubric. Code does not automatically reduce confidence or assign a neutral score.

The final Qwen truth score remains:

```python
qwen_truth_score = (
    confidence
    if verdict == "Real"
    else 100.0 - confidence
)
```

The calibrated fusion formula remains:

```text
V_fac = alpha * S_semantic + (1 - alpha) * S_qwen
```

For a one-round sample, `S_qwen` comes from round one. For a two-round sample, it comes from the cumulative final verdict and confidence returned by round two. `S_semantic` always comes from the single Bocha round.

Because the evidence flow changes, `alpha`, the classification threshold, and temperature calibration must be recalibrated on the development set rather than copied blindly from the current run.

## Failure Handling

- If Bocha fails, preserve the existing public interface and record the failure explicitly. The implementation plan must specify the compatibility fallback without changing label or confidence semantics.
- If Qwen round one fails or cannot be parsed, allow round two to act as a recovery attempt.
- If Qwen round two fails, fall back to the valid round-one verdict and confidence without modifying them.
- If no Qwen round is valid, preserve the current technical-failure fallback behavior and log `verification_failed`.
- API and parse failures are distinct from evidence insufficiency.

## State and Compatibility

Keep the public `InvestigatorOutput` fields used by Judge, Excel output, and evaluation code:

- `label`
- `confidence`
- `evidence`
- `semantic_score`
- `qwen_label`
- `qwen_confidence`
- `iterations`

Add optional internal state for auditability:

- Round-one query and evidence.
- Round-two query and evidence.
- Unverified points.
- Stop reason.
- Whether the verdict changed.
- Retained and rejected evidence.
- Source references when available.

Existing downstream labels and EFI behavior remain unchanged.

## Implementation Scope

The implementation is limited to Investigator-related boundaries:

- `efi_pilot/agents/investigator.py`
  - Make Bocha one-shot.
  - Move the adaptive loop to Qwen.
  - Add the evidence-sufficiency gate and cumulative round state.
  - Remove Bocha query reformulation and cross-round coverage-based selection.
- `efi_pilot/prompts/investigator.py`
  - Preserve the current Bocha query prompt.
  - Extend the first-round Qwen response contract.
  - Add a cumulative second-round Qwen prompt.
- `efi_pilot/agents/base.py`
  - Add only optional audit fields needed by the new loop while preserving consumers.
- `efi_pilot/orchestrator.py`
  - Record both rounds, unverified points, evidence changes, stop reason, and verdict changes.

Arbiter, Judge, Linguist, EFI labels, and the external three-class HD interface are out of scope.

## Verification Plan

### Control Flow

Verify that:

- Sufficient first-round evidence causes exactly one Qwen call.
- Insufficient first-round evidence causes exactly two Qwen calls.
- Bocha is called exactly once per Investigator run.
- Qwen is never called more than twice.
- A failed second round falls back to the unmodified valid first-round result.
- A valid second round evaluates accumulated evidence rather than only its new evidence.

### Known Error Cases

Re-run the 10 known path-conflict errors from `investigator_dev50_rerun_scores.csv`:

- Check whether the nine real texts incorrectly labeled `Factuality` are corrected by verifying the first-round allegations.
- Check whether `CD36/text4` is corrected by searching its remaining unverified details.
- Confirm that a correction is supported by cumulative evidence rather than a blind label reversal.

### Regression Cases

Sample from the 115 conflict-but-correct cases and verify that the loop does not frequently reverse correct `Factuality` decisions to `Real`.

### Ablations

Compare:

1. Current: adaptive Bocha rounds plus repeated independent Qwen verification.
2. One-shot: single Bocha round plus single Qwen round.
3. Proposed: single Bocha round plus evidence-sufficiency-driven Qwen rounds.
4. Fixed-2: single Bocha round plus two Qwen rounds for every sample.

Report:

- Real F1, FacHal F1, and Macro-F1.
- Second-round trigger rate.
- Average Qwen calls per text.
- Average latency.
- Verdict-change count.
- Corrected errors and harmful reversals.

## Success Criteria

- Macro-F1 does not fall below the current calibrated system.
- The proposed loop corrects some of the 10 known errors.
- Corrected errors outnumber harmful reversals.
- Average Qwen calls are materially lower than fixed two-round verification.
- Bocha is called exactly once per text.
- Existing Judge, Excel, and evaluator consumers remain compatible.
