/**
 * Disease knowledge base access — SERVER ONLY.
 *
 * Read at BUILD TIME from the same app/data/diseases.json the bot uses, so the
 * website and the LINE card can never describe a disease differently. There is
 * no second copy of this content to keep in sync.
 *
 * Client components must import types from lib/types.ts instead: pulling this
 * module into the browser bundle would drag `node:fs` with it.
 */
import fs from "node:fs";
import path from "node:path";

import type { Disease } from "./types";

export type { Disease, Severity, ChemicalOption, Reference } from "./types";
export { SEVERITY_LABEL_TH, TYPE_LABEL_TH, PART_FILTERS } from "./types";

const KB_PATH = path.join(process.cwd(), "..", "app", "data", "diseases.json");

export function getAllDiseases(): Disease[] {
  const raw = JSON.parse(fs.readFileSync(KB_PATH, "utf-8"));
  return raw.diseases as Disease[];
}

/** Everything except the "Healthy" entry — that is a state, not a disease. */
export function getBrowsableDiseases(): Disease[] {
  return getAllDiseases().filter((d) => d.type !== "healthy");
}

export function getDiseaseBySlug(slug: string): Disease | undefined {
  return getAllDiseases().find((d) => d.slug === slug);
}
