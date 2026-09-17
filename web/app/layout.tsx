import type { Metadata, Viewport } from "next";
import Link from "next/link";
import { Logo } from "@/components/Logo";
import "./globals.css";

export const metadata: Metadata = {
  title: "หมอทุเรียน — ตรวจโรคทุเรียนง่าย ๆ ผ่าน LINE",
  description:
    "ถ่ายรูปใบ กิ่ง ลำต้น หรือผลทุเรียนที่สงสัย ส่งเข้า LINE แล้วรู้ผลเบื้องต้นทันที " +
    "พร้อมคลังความรู้โรคทุเรียนสำหรับชาวสวน",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#1F5F3A",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="th">
      <head>
        {/* Plain <link> rather than next/font: keeps the build offline-safe
            and lets the browser fetch fonts directly. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Kanit:wght@400;600&family=Sarabun:wght@400;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <header className="site-header">
          <div className="container">
            <Link className="brand" href="/">
              <Logo size={36} />
              หมอทุเรียน
            </Link>
            <nav className="site-nav" aria-label="เมนูหลัก">
              <Link href="/diseases/">คลังความรู้</Link>
              <Link href="/how-to-photo/">วิธีถ่ายรูป</Link>
              <Link href="/about/">เกี่ยวกับเรา</Link>
            </nav>
          </div>
        </header>

        <main>{children}</main>

        <footer className="site-footer">
          <div className="container">
            <p>
              หมอทุเรียน — เครื่องมือช่วยประเมินโรคทุเรียนเบื้องต้นด้วย AI
              สำหรับเกษตรกรชาวสวนทุเรียน
            </p>
            <p className="disclaimer" style={{ color: "#ffffffcc" }}>
              ผลการตรวจเป็นการประเมินเบื้องต้น ไม่ใช่คำวินิจฉัยของผู้เชี่ยวชาญ
              ควรปรึกษาเจ้าหน้าที่เกษตรก่อนใช้สารเคมีเสมอ
            </p>
            <p>
              <Link href="/privacy/">นโยบายความเป็นส่วนตัว</Link>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
