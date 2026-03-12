# Fusion Weighting Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vectorWeight`/`bm25Weight` config parameters actually affect hybrid fusion scoring in the retriever.

**Architecture:** Remove the redundant `vectorScore + 15% bonus` branch from the 3-way `Math.max` in `fuseResults`, making the weighted fusion formula (`vectorScore * vectorWeight + bm25Score * bm25Weight`) the primary scoring path. Preserve the BM25 high-score floor (`>= 0.75 → score * 0.92`) for exact keyword matches.

**Tech Stack:** TypeScript, Node.js test runner, LanceDB

**Spec:** `docs/superpowers/specs/2026-03-11-fusion-weighting-fix-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/retriever.ts` | Modify (lines 540-553) | Change fusion formula in `fuseResults` method |
| `test/retriever-rerank-regression.mjs` | Modify (append) | Add weight-effectiveness and BM25 floor tests |

---

## Chunk 1: Implementation

### Task 1: Create feature branch

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b fix/fusion-weighting-130
```

- [ ] **Step 2: Verify branch**

Run: `git branch --show-current`
Expected: `fix/fusion-weighting-130`

---

### Task 2: Write failing tests for weight effectiveness

**Files:**
- Modify: `test/retriever-rerank-regression.mjs` (append at end of file)

- [ ] **Step 1: Write the failing test — different weights produce different scores**

Append to `test/retriever-rerank-regression.mjs`:

```javascript
// ============================================================================
// Weight effectiveness tests (Issue #130 Layer 3)
// ============================================================================

const weightTestEntry = {
  id: "weight-test-1",
  text: "User prefers dark mode for all applications.",
  vector: [0.6, 0.8],
  category: "preference",
  scope: "global",
  importance: 0.8,
  timestamp: Date.now(),
  metadata: "{}",
};

const weightTestStore = {
  hasFtsSupport: true,
  async vectorSearch() {
    return [{ entry: weightTestEntry, score: 0.6 }];
  },
  async bm25Search() {
    return [{ entry: weightTestEntry, score: 0.5 }];
  },
  async hasId(id) {
    return id === weightTestEntry.id;
  },
};

const weightTestEmbedder = {
  async embedQuery() {
    return [0.6, 0.8];
  },
};

// Test: vectorWeight=0.9, bm25Weight=0.1
const vectorHeavyRetriever = createRetriever(weightTestStore, weightTestEmbedder, {
  ...DEFAULT_RETRIEVAL_CONFIG,
  filterNoise: false,
  rerank: "none",
  vectorWeight: 0.9,
  bm25Weight: 0.1,
  minScore: 0.0,
  hardMinScore: 0.0,
});

// Test: vectorWeight=0.1, bm25Weight=0.9
const bm25HeavyRetriever = createRetriever(weightTestStore, weightTestEmbedder, {
  ...DEFAULT_RETRIEVAL_CONFIG,
  filterNoise: false,
  rerank: "none",
  vectorWeight: 0.1,
  bm25Weight: 0.9,
  minScore: 0.0,
  hardMinScore: 0.0,
});

const vectorHeavyResults = await vectorHeavyRetriever.retrieve({
  query: "dark mode",
  limit: 5,
});

const bm25HeavyResults = await bm25HeavyRetriever.retrieve({
  query: "dark mode",
  limit: 5,
});

assert.equal(vectorHeavyResults.length, 1, "vectorHeavy: should return 1 result");
assert.equal(bm25HeavyResults.length, 1, "bm25Heavy: should return 1 result");

// Core assertion: different weights must produce different fused scores
const vectorHeavyFused = vectorHeavyResults[0].sources.fused?.score;
const bm25HeavyFused = bm25HeavyResults[0].sources.fused?.score;

assert.ok(vectorHeavyFused !== undefined, "vectorHeavy: fused score must exist");
assert.ok(bm25HeavyFused !== undefined, "bm25Heavy: fused score must exist");
assert.notEqual(
  vectorHeavyFused,
  bm25HeavyFused,
  "different weight configs must produce different fused scores",
);

// vectorScore(0.6) > bm25Score(0.5), so vector-heavy config should score higher
// vectorHeavy: 0.6*0.9 + 0.5*0.1 = 0.59
// bm25Heavy:   0.6*0.1 + 0.5*0.9 = 0.51
assert.ok(
  vectorHeavyFused > bm25HeavyFused,
  `vector-heavy config (${vectorHeavyFused}) should score higher than bm25-heavy (${bm25HeavyFused}) when vectorScore > bm25Score`,
);

console.log("OK: weight effectiveness test passed");
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node test/retriever-rerank-regression.mjs`
Expected: FAIL — because current `Math.max` makes both configs produce the same score (formula A dominates both).

---

### Task 3: Write failing test for BM25 high-score floor preservation

**Files:**
- Modify: `test/retriever-rerank-regression.mjs` (append)

- [ ] **Step 1: Write the failing test — BM25 floor holds regardless of weights**

Append to `test/retriever-rerank-regression.mjs`:

```javascript
// Test: BM25 high-score floor (>= 0.75) must hold regardless of weight config
const floorTestEntry = {
  id: "floor-test-1",
  text: "JINA_API_KEY=sk-test-12345",
  vector: [0.1, 0.9],
  category: "credential",
  scope: "global",
  importance: 1.0,
  timestamp: Date.now(),
  metadata: "{}",
};

const floorTestStore = {
  hasFtsSupport: true,
  async vectorSearch() {
    return [{ entry: floorTestEntry, score: 0.3 }];
  },
  async bm25Search() {
    return [{ entry: floorTestEntry, score: 0.85 }];
  },
  async hasId(id) {
    return id === floorTestEntry.id;
  },
};

const floorTestRetriever = createRetriever(floorTestStore, weightTestEmbedder, {
  ...DEFAULT_RETRIEVAL_CONFIG,
  filterNoise: false,
  rerank: "none",
  vectorWeight: 0.9,
  bm25Weight: 0.1,
  minScore: 0.0,
  hardMinScore: 0.0,
});

const floorResults = await floorTestRetriever.retrieve({
  query: "JINA_API_KEY",
  limit: 5,
});

assert.equal(floorResults.length, 1, "floor test: should return 1 result");

const floorFused = floorResults[0].sources.fused?.score;
assert.ok(floorFused !== undefined, "floor test: fused score must exist");

// BM25 floor: 0.85 * 0.92 = 0.782
// Weighted: 0.3 * 0.9 + 0.85 * 0.1 = 0.355
// Math.max(0.355, 0.782) = 0.782 — floor should win
assert.ok(
  floorFused >= 0.85 * 0.92 - 0.001,
  `BM25 high-score floor must hold: got ${floorFused}, expected >= ${0.85 * 0.92}`,
);

console.log("OK: BM25 high-score floor test passed");
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `node test/retriever-rerank-regression.mjs`
Expected: This test may pass even before the fix (the floor branch already exists). That's fine — it serves as a regression guard.

---

### Task 4: Implement the fusion formula fix

**Files:**
- Modify: `src/retriever.ts` (lines 544-553)

- [ ] **Step 1: Modify fuseResults fusion formula**

In `src/retriever.ts`, replace lines 544-553:

```typescript
      // Base = vector score; BM25 hit boosts by up to 15%
      // BM25-only results use their raw BM25 score so exact keyword matches
      // (e.g. searching "JINA_API_KEY") still surface. The previous floor of 0.5
      // was too generous and allowed ghost entries to survive hardMinScore (0.35).
      const fusedScore = vectorResult
        ? clamp01(
          Math.max(
            vectorScore + (bm25Hit * 0.15 * vectorScore),
            (vectorScore * this.config.vectorWeight) + (bm25Score * this.config.bm25Weight),
            bm25Score >= 0.75 ? bm25Score * 0.92 : 0,
          ),
          0.1,
        )
        : clamp01(bm25Result!.score, 0.1);
```

With:

```typescript
      // Weighted fusion: vectorWeight/bm25Weight directly control score blending.
      // BM25 high-score floor (>= 0.75) preserves exact keyword matches
      // (e.g. API keys, ticket numbers) that may have low vector similarity.
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

- [ ] **Step 2: Run all tests to verify they pass**

Run: `node test/retriever-rerank-regression.mjs`
Expected: ALL tests pass — existing regression tests + new weight effectiveness + BM25 floor test.

---

### Task 5: Run full test suite

- [ ] **Step 1: Run the project's full test suite**

Run: `npm test`
Expected: All tests pass. If any test fails, investigate whether it's related to the fusion change.

---

### Task 6: Commit implementation

- [ ] **Step 1: Stage and commit**

```bash
git add src/retriever.ts test/retriever-rerank-regression.mjs
git commit -m "fix(retriever): make vectorWeight/bm25Weight actually affect fusion scoring

Remove redundant vectorScore + 15% bonus branch from fuseResults Math.max.
Weighted fusion (vectorScore * vectorWeight + bm25Score * bm25Weight) is now
the primary scoring formula. BM25 high-score floor (>= 0.75) preserved for
exact keyword matches.

Closes #130 (Layer 3)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 2: Verify commit**

Run: `git log --oneline -3`
Expected: New commit appears at HEAD.
