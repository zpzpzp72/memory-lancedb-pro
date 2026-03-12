# Fix: Make vectorWeight/bm25Weight Actually Affect Fusion Scoring

**Issue:** [#130](https://github.com/CortexReach/memory-lancedb-pro/issues/130) (Layer 3)
**Date:** 2026-03-11
**Author:** ssyn0813

## Problem

In `src/retriever.ts`, the `fuseResults` method uses `Math.max` to pick the highest score from three competing formulas:

```typescript
const fusedScore = vectorResult
  ? clamp01(
      Math.max(
        vectorScore + (bm25Hit * 0.15 * vectorScore),           // A: 15% bonus
        (vectorScore * this.config.vectorWeight) + (bm25Score * this.config.bm25Weight),  // B: weighted
        bm25Score >= 0.75 ? bm25Score * 0.92 : 0,               // C: BM25 floor
      ),
      0.1,
    )
  : clamp01(bm25Result!.score, 0.1);
```

Formula A (`vectorScore + 15% bonus`) dominates in most cases because `vectorScore` is typically > 0.5, making the weighted fusion (formula B) rarely win. As a result, users cannot meaningfully tune retrieval behavior by adjusting `vectorWeight`/`bm25Weight`.

## Solution (Approach B — Weighted Fusion + BM25 Floor)

Remove the redundant formula A. Make weighted fusion the primary scoring formula while preserving the BM25 high-score floor for exact keyword matches.

```typescript
const weightedFusion = (vectorScore * this.config.vectorWeight)
                     + (bm25Score * this.config.bm25Weight);
const fusedScore = vectorResult
  ? clamp01(
      Math.max(
        weightedFusion,
        bm25Score >= 0.75 ? bm25Score * 0.92 : 0,
      ),
      0.1,
    )
  : clamp01(bm25Result!.score, 0.1);
```

### Why this approach

- **Weights take effect** — `vectorWeight`/`bm25Weight` directly control the fusion formula
- **Safe** — BM25 high-score floor (`>= 0.75 → score * 0.92`) preserves exact keyword matches (API keys, ticket numbers)
- **BM25-only path unchanged** — entries with no vector result still use raw BM25 score
- **Backward compatible** — existing regression tests pass with no modifications

### Approaches considered but rejected

| Approach | Why rejected |
|----------|-------------|
| A: Pure weighted fusion (no floor) | Exact keyword matches could be dragged down by low vectorScore |
| C: Keep 3-way Math.max, reduce bonus to 5% | Doesn't fundamentally fix the problem; formula A still competes |

## Numerical Verification

Default config: `vectorWeight=0.7`, `bm25Weight=0.3`, `hardMinScore=0.35`

Rows 1-2 use the test-specific override `hardMinScore=0.62` (from `retriever-rerank-regression.mjs`). Rows 3-4 use the true default `hardMinScore=0.35`.

| Scenario | vectorScore | bm25Score | Old formula | New formula | hardMinScore | Passes? |
|----------|------------|-----------|-------------|-------------|-------------|---------|
| Rerank regression test | 0.5439 | 0.7834 | 0.721 | 0.721 | 0.62 | Yes |
| Lexical test (乌龙茶) | 0.5007 | 0.78 | 0.718 | 0.718 | 0.62 | Yes |
| High vector, no BM25 | 0.9 | 0.0 | 0.9 | 0.63 | 0.35 | Yes |
| Balanced scores | 0.6 | 0.5 | 0.69 | 0.57 | 0.35 | Yes |

Row 3 shows the expected behavior change: pure vector results are no longer inflated beyond what the weight config dictates.

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/retriever.ts` | Modify `fuseResults` fusion formula | ~5 |
| `test/retriever-rerank-regression.mjs` | Add weight-effectiveness tests | ~60 |

## Test Plan

1. **Weight effectiveness** — same input with different weight configs produces different fusion scores
2. **BM25 high-score floor** — bm25Score >= 0.75 always floors at `bm25Score * 0.92` regardless of weights
3. **Existing regression** — all existing assertions in `retriever-rerank-regression.mjs` pass unchanged

## Scope Exclusions

- Layer 1 (keywords/aliases fields for multilingual BM25) — future PR
- Layer 2 (query expansion) — future PR
