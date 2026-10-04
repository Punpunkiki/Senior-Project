export const metadata = {
  title: "วิธีถ่ายรูปให้ตรวจแม่น — หมอทุเรียน",
  description: "เปรียบเทียบรูปที่ถ่ายดีกับรูปที่ถ่ายไม่ดี เพื่อให้ AI ตรวจโรคได้แม่นขึ้น",
};

const GOOD = [
  "ถ่ายใกล้ ๆ ให้เห็นรอยโรคเต็มกรอบ",
  "ถ่ายกลางแจ้งหรือที่มีแสงธรรมชาติ",
  "โฟกัสชัด ไม่สั่น ไม่เบลอ",
  "ถ่ายทีละจุด ทีละใบ",
];

const BAD = [
  "ถ่ายไกลจนมองไม่เห็นรอย",
  "ถ่ายในที่มืดหรือมีเงาบัง",
  "ภาพเบลอเพราะมือสั่น",
  "ถ่ายรวมหลายใบจนไม่รู้ว่าดูใบไหน",
];

export default function HowToPhotoPage() {
  return (
    <section>
      <div className="container">
        <h1>วิธีถ่ายรูปให้ตรวจแม่น</h1>
        <p>
          หมอดูจากรูปอย่างเดียว ถ้ารูปชัดและมีแสงพอ
          ผลตรวจก็จะแม่นขึ้นมาก
        </p>

        <div className="card-grid">
          <article className="card" style={{ borderColor: "var(--leaf-500)" }}>
            <h2 style={{ color: "var(--leaf-500)" }}>✅ ถ่ายแบบนี้ดี</h2>
            <ul className="tick-list">
              {GOOD.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </article>

          <article className="card" style={{ borderColor: "var(--danger-600)" }}>
            <h2 style={{ color: "var(--danger-600)" }}>❌ แบบนี้ตรวจยาก</h2>
            <ul className="tick-list">
              {BAD.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </article>
        </div>

        <div className="notice">
          <p>
            ถ้าไม่แน่ใจ ถ่ายมาหลายมุมแล้วส่งทีละรูปได้
            หมอจะตรวจให้ทีละรูปเลย
          </p>
        </div>
      </div>
    </section>
  );
}
