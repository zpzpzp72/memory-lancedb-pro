export type ReflectionKind = "inherit" | "derive";

export function parseReflectionMetadata(metadataRaw: string | undefined): Record<string, unknown> {
  if (!metadataRaw) return {};
  try {
    const parsed = JSON.parse(metadataRaw);
    return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

export function getReflectionKind(metadata: Record<string, unknown>): ReflectionKind | undefined {
  const kindRaw = typeof metadata.reflectionKind === "string" ? metadata.reflectionKind.trim().toLowerCase() : "";
  if (kindRaw === "inherit" || kindRaw === "derive") return kindRaw;
  return undefined;
}

export function isReflectionEntry(entry: { category: string; metadata?: string }): boolean {
  if (entry.category === "reflection") return true;
  const metadata = parseReflectionMetadata(entry.metadata);
  return metadata.type === "memory-reflection";
}

export function getDisplayCategoryTag(entry: { category: string; scope: string; metadata?: string }): string {
  if (!isReflectionEntry(entry)) return `${entry.category}:${entry.scope}`;
  const metadata = parseReflectionMetadata(entry.metadata);
  const kind = getReflectionKind(metadata);
  if (kind === "inherit") return "reflection:Inherit";
  if (kind === "derive") return "reflection:Derive";
  return `reflection:${entry.scope}`;
}
