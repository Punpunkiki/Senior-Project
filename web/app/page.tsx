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
    body: "กดปุ่มเพิ่มเพื่อนด้านบน แล้วเปิดแชทหมอทุเรียนได้เลย",
  },
  {
    title: "ถ่ายรูปแล้วส่งเข้าแชทได้เลย",
    body: "ถ่ายใบ กิ่ง ลำต้น หรือผลที่ดูผิดปกติ ให้เห็นรอยชัด ๆ ไม่ต้องกดเมนูอะไรก่อน",
  },
  {
    title: "รู้ผลเบื้องต้นทันที",
    body: "ระบบจะบอกว่าน่าจะเป็นอะไร อาการ สาเหตุ และต้องรีบจัดการหรือแค่เฝ้าระวัง",
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
            ถ่ายรูปส่วนที่ดูผิดปกติส่งเข้า LINE แล้วรู้ผลเบื้องต้นทันที
            ไม่ต้องติดตั้งแอปเพิ่ม
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
          <h2>ดูสิ่งผิดปกติที่พบได้บนต้นทุเรียน</h2>
          <p>
            รวมทั้งที่เกิดจากเชื้อโรค แมลง และอาการผิดปกติที่ไม่ได้เกิดจากเชื้อ
            แต่ละรายการบอกไว้ว่าเจอแล้วต้องรีบจัดการ หรือแค่เฝ้าระวังต่อ
          </p>
          <Link className="btn btn-primary" href="/diseases/">
            📚 เปิดคลังความรู้
          </Link>
        </div>
      </section>
    </>
  );
}
