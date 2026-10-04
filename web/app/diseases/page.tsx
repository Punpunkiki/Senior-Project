import { getBrowsableDiseases } from "@/lib/diseases";
import { DiseaseList } from "./DiseaseList";

export const metadata = {
  title: "คลังความรู้สิ่งผิดปกติบนทุเรียน — หมอทุเรียน",
  description:
    "รวมสิ่งผิดปกติที่พบบนต้นทุเรียน ทั้งโรค แมลง และอาการผิดปกติ " +
    "พร้อมบอกว่าเจอแล้วต้องทำอะไรต่อ",
};

export default function DiseasesPage() {
  // Read at build time, then hand to a client component for search.
  const diseases = getBrowsableDiseases();
  return (
    <section>
      <div className="container">
        <h1>คลังความรู้สิ่งผิดปกติบนทุเรียน</h1>
        <p>
          รวมสิ่งผิดปกติที่พบได้บนต้นทุเรียน ทั้งที่เกิดจากเชื้อโรค แมลง
          และอาการผิดปกติที่ไม่ได้เกิดจากเชื้อ แต่ละรายการจะบอกไว้ด้วยว่า
          เจอแล้วต้องรีบจัดการ หรือแค่เฝ้าระวังต่อ
        </p>
        <DiseaseList diseases={diseases} />
      </div>
    </section>
  );
}
