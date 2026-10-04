/**
 * Shared types and label maps — deliberately free of any Node import so that
 * client components can use them. The filesystem access that reads the
 * knowledge base lives in lib/diseases.ts, which is server-only.
 *
 * Wording note: the site talks about "สิ่งผิดปกติ" (abnormalities found on a
 * durian tree), not "โรค" (diseases). Several entries are insects and one is
 * a symptom with no pathogen at all, so "โรค" was simply wrong for them.
 */

export type Severity = "healthy" | "watch" | "moderate" | "severe";

export interface ChemicalOption {
  active_ingredient: string;
  note: string;
}

export interface Reference {
  title: string;
  url: string;
}

export interface Disease {
  class_name: string;
  slug: string;
  name_th: string;
  name_en: string;
  pathogen: string | null;
  type: string | null;
  severity: Severity;
  affected_parts: string[];
  symptoms: string[];
  causes_conditions: string[];
  immediate_actions: string[];
  chemical_options: ChemicalOption[];
  prevention: string[];
  when_to_call_expert: string;
  related_note: string;
  images: string[];
  references: Reference[];
  reviewed_by_expert: boolean;
  pending_expert_input: boolean;
}

export const SEVERITY_LABEL_TH: Record<Severity, string> = {
  healthy: "ปกติ",
  watch: "เฝ้าระวัง",
  moderate: "ควรจัดการ",
  severe: "รุนแรง",
};

/**
 * Calling something "ผิดปกติ" is only useful if the reader is also told what
 * that means for them today. Every entry therefore carries an explicit next
 * step, derived from its severity so it can never be forgotten.
 */
export const NEXT_ACTION_TH: Record<Severity, string> = {
  healthy: "ดูแลตามปกติ ไม่ต้องทำอะไรเพิ่ม",
  watch: "ยังไม่ต้องใช้สาร ให้คอยสังเกตอาการต่อ ถ้าลามขึ้นค่อยจัดการ",
  moderate: "ควรลงมือจัดการภายในสัปดาห์นี้ อย่าปล่อยทิ้งไว้",
  severe: "ต้องรีบจัดการทันที ถ้าช้าอาจเสียทั้งกิ่งหรือทั้งต้น",
};

export const TYPE_LABEL_TH: Record<string, string> = {
  fungus: "เชื้อรา",
  oomycete: "ราน้ำ",
  algae: "สาหร่าย",
  insect: "แมลง",
  disorder: "อาการผิดปกติ (ไม่ใช่เชื้อโรค)",
  healthy: "ปกติ",
};

/**
 * The knowledge base is written for the LINE bot, whose persona ends every
 * sentence with "ครับ". On the website that reads as chat rather than
 * reference material, so the particle is dropped at render time and the
 * single source of content stays shared with the bot.
 */
export function stripPolite(text: string): string {
  return text
    .replace(/\s*ครับ(?=[\s.,)]|$)/g, "")
    .replace(/\s+([,.])/g, "$1")
    .trim();
}
