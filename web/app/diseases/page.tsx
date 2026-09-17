import { getBrowsableDiseases } from "@/lib/diseases";
import { DiseaseList } from "./DiseaseList";

export const metadata = {
  title: "คลังความรู้โรคทุเรียน — หมอทุเรียน",
  description:
    "รวมโรคและแมลงศัตรูทุเรียน อาการ สาเหตุ วิธีแก้ไข และวิธีป้องกัน",
};

export default function DiseasesPage() {
  // Read at build time, then hand to a client component for search/filter.
  const diseases = getBrowsableDiseases();
  return (
    <section>
      <div className="container">
        <h1>คลังความรู้โรคทุเรียน</h1>
        <p>
          เลือกดูรายละเอียดแต่ละโรคได้เลยครับ ค้นหาด้วยชื่อโรค
          หรือกรองตามส่วนของต้นที่พบอาการ
        </p>
        <DiseaseList diseases={diseases} />
      </div>
    </section>
  );
}
