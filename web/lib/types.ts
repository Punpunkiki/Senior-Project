/**
 * Shared types and label maps — deliberately free of any Node import so that
 * client components can use them. The filesystem access that reads the
 * knowledge base lives in lib/diseases.ts, which is server-only.
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

export const TYPE_LABEL_TH: Record<string, string> = {
  fungus: "เชื้อรา",
  oomycete: "ราน้ำ",
  algae: "สาหร่าย",
  insect: "แมลง",
  disorder: "อาการผิดปกติ (ไม่ใช่เชื้อโรค)",
  healthy: "ปกติ",
};

/** Groups used by the /diseases filter chips. */
export const PART_FILTERS = ["ใบ", "ดอก", "กิ่ง", "ลำต้น", "ราก", "ผล"];
