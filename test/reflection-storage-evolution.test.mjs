import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

const {
  storeReflectionToLanceDB,
  loadAgentReflectionSlicesFromEntries,
  REFLECTION_DERIVE_LOGISTIC_K,
  REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
  REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT,
} = jiti("../src/reflection-store.ts");

function makeEntry({ timestamp, metadata, category = "reflection", scope = "global" }) {
  return {
    id: `mem-${Math.random().toString(36).slice(2, 8)}`,
    text: "reflection-entry",
    vector: [],
    category,
    scope,
    importance: 0.7,
    timestamp,
    metadata: JSON.stringify(metadata),
  };
}

describe("reflection split persistence", () => {
  it("stores split inherit/derive entries with subtype metadata and logistic fields", async () => {
    const storedEntries = [];
    const vectorSearchCalls = [];

    const result = await storeReflectionToLanceDB({
      reflectionText: [
        "## Invariants",
        "- Always confirm assumptions before changing files.",
        "## Derived",
        "- Next run verify reflection split with targeted tests.",
      ].join("\n"),
      sessionKey: "agent:main:session:abc",
      sessionId: "abc",
      agentId: "main",
      command: "command:reset",
      scope: "global",
      toolErrorSignals: [{ signatureHash: "deadbeef" }],
      runAt: 1_700_000_000_000,
      usedFallback: false,
      embedPassage: async (text) => [text.length],
      vectorSearch: async (vector) => {
        vectorSearchCalls.push(vector);
        return [];
      },
      store: async (entry) => {
        storedEntries.push(entry);
        return { ...entry, id: `id-${storedEntries.length}`, timestamp: 1_700_000_000_000 };
      },
    });

    assert.equal(result.stored, true);
    assert.deepEqual(new Set(result.storedKinds), new Set(["inherit", "derive"]));
    assert.equal(storedEntries.length, 2);
    assert.equal(vectorSearchCalls.length, 2);

    const inherit = storedEntries.find((entry) => JSON.parse(entry.metadata).reflectionKind === "inherit");
    const derive = storedEntries.find((entry) => JSON.parse(entry.metadata).reflectionKind === "derive");
    assert.ok(inherit);
    assert.ok(derive);

    const inheritMeta = JSON.parse(inherit.metadata);
    const deriveMeta = JSON.parse(derive.metadata);

    assert.equal(inherit.category, "reflection");
    assert.equal(derive.category, "reflection");
    assert.deepEqual(inheritMeta.invariants, ["Always confirm assumptions before changing files."]);
    assert.equal(inheritMeta.reflectionKind, "inherit");

    assert.equal(deriveMeta.reflectionKind, "derive");
    assert.deepEqual(deriveMeta.derived, ["Next run verify reflection split with targeted tests."]);
    assert.equal(deriveMeta.decayModel, "logistic");
    assert.equal(deriveMeta.decayK, REFLECTION_DERIVE_LOGISTIC_K);
    assert.equal(deriveMeta.decayMidpointDays, REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS);
    assert.equal(deriveMeta.deriveBaseWeight, 1);
  });
});

describe("reflection legacy compatibility + source separation", () => {
  it("loads legacy combined entries and separates inherit/derive sources", () => {
    const now = Date.UTC(2026, 2, 7);

    const entries = [
      makeEntry({
        timestamp: now - 30 * 60 * 1000,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          reflectionKind: "inherit",
          invariants: ["Always keep fixes minimal."],
          derived: ["should-not-appear-from-inherit"],
          storedAt: now - 30 * 60 * 1000,
        },
      }),
      makeEntry({
        timestamp: now - 25 * 60 * 1000,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          reflectionKind: "derive",
          invariants: ["should-not-appear-from-derive"],
          derived: ["Next run keep logs concise."],
          storedAt: now - 25 * 60 * 1000,
        },
      }),
      makeEntry({
        timestamp: now - 20 * 60 * 1000,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          invariants: ["Legacy invariant still applies."],
          derived: ["Legacy derived delta still applies."],
          storedAt: now - 20 * 60 * 1000,
        },
      }),
    ];

    const slices = loadAgentReflectionSlicesFromEntries({
      entries,
      agentId: "main",
      now,
      deriveMaxAgeMs: 7 * 24 * 60 * 60 * 1000,
    });

    assert.ok(slices.invariants.includes("Always keep fixes minimal."));
    assert.ok(slices.invariants.includes("Legacy invariant still applies."));
    assert.ok(!slices.invariants.includes("should-not-appear-from-derive"));

    assert.ok(slices.derived.includes("Next run keep logs concise."));
    assert.ok(slices.derived.includes("Legacy derived delta still applies."));
    assert.ok(!slices.derived.includes("should-not-appear-from-inherit"));
  });
});

describe("reflection derive logistic scoring", () => {
  it("aggregates multiple recent derive memories and down-weights fallback entries", () => {
    const now = Date.UTC(2026, 2, 7);
    const day = 24 * 60 * 60 * 1000;

    const entries = [
      makeEntry({
        timestamp: now - 1 * day,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          reflectionKind: "derive",
          derived: ["Fresh normal derive"],
          storedAt: now - 1 * day,
          deriveBaseWeight: 1,
          usedFallback: false,
        },
      }),
      makeEntry({
        timestamp: now - 1 * day,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          reflectionKind: "derive",
          derived: ["Fresh fallback derive"],
          storedAt: now - 1 * day,
          usedFallback: true,
        },
      }),
      makeEntry({
        timestamp: now - 5 * day,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          reflectionKind: "derive",
          derived: ["Older normal derive"],
          storedAt: now - 5 * day,
          usedFallback: false,
        },
      }),
      makeEntry({
        timestamp: now - 2 * day,
        metadata: {
          type: "memory-reflection",
          agentId: "main",
          reflectionKind: "derive",
          derived: ["Second recent derive signal"],
          storedAt: now - 2 * day,
          usedFallback: false,
        },
      }),
    ];

    const slices = loadAgentReflectionSlicesFromEntries({
      entries,
      agentId: "main",
      now,
      deriveMaxAgeMs: 10 * day,
    });

    assert.equal(slices.derived[0], "Fresh normal derive");
    assert.ok(slices.derived.includes("Second recent derive signal"));

    const fallbackIdx = slices.derived.indexOf("Fresh fallback derive");
    const olderIdx = slices.derived.indexOf("Older normal derive");
    assert.notEqual(fallbackIdx, -1);
    assert.notEqual(olderIdx, -1);
    assert.ok(fallbackIdx < olderIdx, "fallback should still rank above much older derive due recency, but below normal fresh derive");

    assert.equal(REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT, 0.35);
  });
});
