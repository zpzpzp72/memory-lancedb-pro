import { describe, it } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { getDisplayCategoryTag } = jiti("../src/reflection-metadata.ts");

describe("reflection display category tags", () => {
  it("shows subtype labels for new reflection entries", () => {
    assert.equal(
      getDisplayCategoryTag({
        category: "reflection",
        scope: "global",
        metadata: JSON.stringify({ type: "memory-reflection", reflectionKind: "inherit" }),
      }),
      "reflection:Inherit"
    );

    assert.equal(
      getDisplayCategoryTag({
        category: "reflection",
        scope: "global",
        metadata: JSON.stringify({ type: "memory-reflection", reflectionKind: "derive" }),
      }),
      "reflection:Derive"
    );
  });

  it("keeps legacy-safe reflection display path when subtype metadata is absent", () => {
    assert.equal(
      getDisplayCategoryTag({
        category: "reflection",
        scope: "project-a",
        metadata: JSON.stringify({ type: "memory-reflection", invariants: ["Always verify output"] }),
      }),
      "reflection:project-a"
    );
  });

  it("preserves non-reflection display categories", () => {
    assert.equal(
      getDisplayCategoryTag({
        category: "fact",
        scope: "global",
        metadata: "{}",
      }),
      "fact:global"
    );
  });
});
