import { Suspense } from "react";
import { ResultView } from "./ResultView";

export const metadata = {
  title: "ผลตรวจโรคทุเรียน — หมอทุเรียน",
};

/**
 * The LIFF result page.
 *
 * Uses a query parameter (/result?dg=<id>) rather than a dynamic segment
 * because this site is a static export: diagnosis ids are created at runtime
 * and cannot be enumerated at build time. LIFF forwards query parameters from
 * https://liff.line.me/<liffId>?dg=<id> to this endpoint.
 */
export default function ResultPage() {
  return (
    <section>
      <div className="container">
        <Suspense fallback={<p>กำลังโหลดผลตรวจ…</p>}>
          <ResultView />
        </Suspense>
      </div>
    </section>
  );
}
