import Link from "next/link";
import { notFound } from "next/navigation";
import {
  getAllDiseases,
  getBrowsableDiseases,
  getDiseaseBySlug,
  TYPE_LABEL_TH,
} from "@/lib/diseases";
import { Disclaimer, ReviewedBadge, SeverityBadge } from "@/components/ui";

/** Static export needs every slug up front. */
export function generateStaticParams() {
  return getAllDiseases().map((d) => ({ slug: d.slug }));
}

type Props = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: Props) {
  const { slug } = await params;
  const disease = getDiseaseBySlug(slug);
  if (!disease) return { title: "ไม่พบข้อมูลโรค — หมอทุเรียน" };
  return {
    title: `${disease.name_th} — หมอทุเรียน`,
    description:
      disease.symptoms[0] ?? `ข้อมูลโรค ${disease.name_th} สำหรับชาวสวนทุเรียน`,
  };
}

function Section({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <>
      <h2>{title}</h2>
      <ul className="tick-list">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </>
  );
}

export default async function DiseasePage({ params }: Props) {
  const { slug } = await params;
  const disease = getDiseaseBySlug(slug);
  if (!disease) notFound();

  const related = getBrowsableDiseases()
    .filter((d) => d.slug !== disease.slug && d.type === disease.type)
    .slice(0, 3);

  return (
    <section>
      <div className="container">
        <p>
          <Link href="/diseases/">← กลับไปคลังความรู้</Link>
        </p>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap",
                      marginBottom: 12 }}>
          <SeverityBadge severity={disease.severity}
                         pending={disease.pending_expert_input} />
          <ReviewedBadge reviewed={disease.reviewed_by_expert} />
        </div>

        <h1>{disease.name_th}</h1>
        <p className="muted">
          {disease.name_en}
          {disease.pathogen ? ` · ${disease.pathogen}` : ""}
          {disease.type ? ` · ${TYPE_LABEL_TH[disease.type] ?? disease.type}` : ""}
        </p>

        {disease.affected_parts.length > 0 && (
          <p>
            <strong>ส่วนที่พบอาการ:</strong> {disease.affected_parts.join(", ")}
          </p>
        )}

        {disease.pending_expert_input ? (
          <div className="notice">
            <p>
              ข้อมูลคำแนะนำของโรคนี้กำลังจัดทำและรอผู้เชี่ยวชาญตรวจทานอยู่ครับ
              ระหว่างนี้แนะนำให้ปรึกษาเจ้าหน้าที่เกษตรโดยตรงครับ
            </p>
          </div>
        ) : (
          <>
            {!disease.reviewed_by_expert && (
              <div className="notice">
                <p>
                  เนื้อหานี้ยังอยู่ระหว่างรอนักวิชาการเกษตรตรวจทาน
                  โปรดใช้เป็นข้อมูลเบื้องต้นเท่านั้นครับ
                </p>
              </div>
            )}

            <Section title="🔍 อาการที่พบ" items={disease.symptoms} />
            <Section title="🦠 สาเหตุและสภาพที่ทำให้ระบาด"
                     items={disease.causes_conditions} />
            <Section title="🛠️ วิธีแก้ไขทันที" items={disease.immediate_actions} />

            {disease.chemical_options.length > 0 && (
              <>
                <h2>🧪 สารที่ใช้ได้</h2>
                <ul className="tick-list">
                  {disease.chemical_options.map((option) => (
                    <li key={option.active_ingredient}>
                      <strong>{option.active_ingredient}</strong> — {option.note}
                    </li>
                  ))}
                </ul>
                <div className="notice notice-danger">
                  <p>
                    อ่านฉลากและใช้ตามอัตราที่ระบุบนฉลากเสมอ
                    และปรึกษาเจ้าหน้าที่เกษตรก่อนใช้สารเคมีครับ
                  </p>
                </div>
              </>
            )}

            <Section title="🛡️ วิธีป้องกัน" items={disease.prevention} />

            {disease.when_to_call_expert && (
              <>
                <h2>📞 เมื่อไหร่ควรเรียกผู้เชี่ยวชาญ</h2>
                <p>{disease.when_to_call_expert}</p>
              </>
            )}

            {disease.references.length > 0 && (
              <>
                <h2>📚 แหล่งอ้างอิง</h2>
                <ul className="tick-list">
                  {disease.references.map((ref) => (
                    <li key={ref.url}>
                      <a href={ref.url} target="_blank" rel="noreferrer">
                        {ref.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}

        <Disclaimer />

        {related.length > 0 && (
          <>
            <h2>โรคใกล้เคียง</h2>
            <ul className="tick-list">
              {related.map((r) => (
                <li key={r.slug}>
                  <Link href={`/diseases/${r.slug}/`}>{r.name_th}</Link>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </section>
  );
}
