import Link from "next/link";
import { Logo } from "@/components/Logo";

/** LINE friend-add link. Set NEXT_PUBLIC_LINE_ID at build time. */
const LINE_ID = process.env.NEXT_PUBLIC_LINE_ID ?? "";
const addFriendUrl = LINE_ID
  ? `https://line.me/R/ti/p/${encodeURIComponent(LINE_ID)}`
  : "";

const STEPS = [
  {
    title: "เพิ่มเพื่อนใน LINE",
    body: "กดปุ่มเพิ่มเพื่อนด้านบน แล้วเปิดแชทหมอทุเรียนได้เลยครับ",
  },
  {
    title: "กดปุ่ม “ตรวจโรคทุเรียน” แล้วถ่ายรูป",
    body: "ถ่ายใบ กิ่ง ลำต้น หรือผลที่สงสัยว่าเป็นโรค ให้เห็นรอยชัด ๆ ครับ",
  },
  {
    title: "รู้ผลเบื้องต้นทันที",
    body: "หมอจะบอกชื่อโรค อาการ สาเหตุ วิธีแก้ไข และวิธีป้องกันให้ครับ",
  },
];

export default function HomePage() {
  return (
    <>
      <section className="hero thorn-bg">
        <div className="container">
          <Logo size={84} />
          <h1>หมอทุเรียน ตรวจโรคทุเรียนง่าย ๆ ผ่าน LINE</h1>
          <p className="lead">
            ถ่ายรูปส่วนที่สงสัยส่งเข้า LINE แล้วรู้ผลเบื้องต้นทันที
            ไม่ต้องติดตั้งแอปเพิ่มครับ
          </p>
          <div className="hero-actions">
            {addFriendUrl ? (
              <a className="btn btn-accent" href={addFriendUrl}>
                ➕ เพิ่มเพื่อนใน LINE
              </a>
            ) : (
              <span className="btn btn-accent" aria-disabled="true">
                ➕ เพิ่มเพื่อนใน LINE
              </span>
            )}
            <Link className="btn btn-secondary" href="/diseases/">
              📚 ดูคลังความรู้
            </Link>
          </div>
        </div>
      </section>

      <section>
        <div className="container">
          <h2>ใช้งานยังไง</h2>
          <ol className="steps">
            {STEPS.map((step, i) => (
              <li className="step" key={step.title}>
                <span className="step-num" aria-hidden="true">{i + 1}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="section-alt">
        <div className="container">
          <h2>อยากให้ตรวจแม่นขึ้น</h2>
          <p>
            รูปที่ชัดและมีแสงพอ ช่วยให้หมอตรวจได้แม่นขึ้นมากครับ
            ดูตัวอย่างรูปที่ถ่ายดีกับรูปที่ถ่ายไม่ดีได้ที่นี่
          </p>
          <Link className="btn btn-primary" href="/how-to-photo/">
            📸 วิธีถ่ายรูปให้ตรวจแม่น
          </Link>
        </div>
      </section>
    </>
  );
}
