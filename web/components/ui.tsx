/** Shared presentational pieces used across pages. */
import Link from "next/link";
import { Disease, SEVERITY_LABEL_TH, Severity } from "@/lib/types";

export function SeverityBadge({ severity, pending }: {
  severity: Severity;
  pending?: boolean;
}) {
  if (pending) {
    return <span className="badge badge-pending">ข้อมูลกำลังจัดทำ</span>;
  }
  return (
    <span className={`badge badge-${severity}`}>
      {SEVERITY_LABEL_TH[severity]}
    </span>
  );
}

export function ReviewedBadge({ reviewed }: { reviewed: boolean }) {
  if (!reviewed) return null;
  return <span className="badge badge-reviewed">✓ ตรวจทานโดยผู้เชี่ยวชาญแล้ว</span>;
}

export function ConfidenceBar({ value, color }: {
  value: number;
  color?: string;
}) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div>
      <div
        style={{ display: "flex", justifyContent: "space-between",
                 fontSize: "var(--text-caption)" }}
      >
        <span className="muted">ความมั่นใจ</span>
        <strong>{pct}%</strong>
      </div>
      <div
        className="conf-track"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`ความมั่นใจ ${pct} เปอร์เซ็นต์`}
      >
        <div
          className="conf-fill"
          style={{ width: `${Math.max(pct, 2)}%`,
                   background: color ?? "var(--leaf-500)" }}
        />
      </div>
    </div>
  );
}

export function DiseaseCard({ disease }: { disease: Disease }) {
  return (
    <article className="card">
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap",
                    marginBottom: 8 }}>
        <SeverityBadge severity={disease.severity}
                       pending={disease.pending_expert_input} />
        <ReviewedBadge reviewed={disease.reviewed_by_expert} />
      </div>
      <h3>{disease.name_th}</h3>
      <p className="card-sub">
        {disease.name_en}
        {disease.pathogen ? ` · ${disease.pathogen}` : ""}
      </p>
      <p>
        {disease.symptoms[0] ??
          "ข้อมูลอาการของโรคนี้กำลังจัดทำและรอผู้เชี่ยวชาญตรวจทานครับ"}
      </p>
      <Link className="btn btn-secondary" href={`/diseases/${disease.slug}/`}>
        อ่านรายละเอียด
      </Link>
    </article>
  );
}

export function Disclaimer() {
  return (
    <p className="disclaimer">
      ผลนี้เป็นการประเมินเบื้องต้นจาก AI ควรปรึกษาเจ้าหน้าที่เกษตรก่อนใช้สารเคมี
      และใช้สารตามอัตราบนฉลากเสมอ
    </p>
  );
}
