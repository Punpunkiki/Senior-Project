"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ConfidenceBar, Disclaimer, SeverityBadge } from "@/components/ui";
import { Severity } from "@/lib/types";

const LIFF_ID = process.env.NEXT_PUBLIC_LIFF_ID ?? "";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

const SEVERITY_COLOR: Record<Severity, string> = {
  healthy: "var(--leaf-500)",
  watch: "var(--durian-500)",
  moderate: "var(--warn-500)",
  severe: "var(--danger-600)",
};

const TIER_TITLE: Record<string, string> = {
  confident: "ผลตรวจ",
  uncertain_top2: "อาจเป็นโรคนี้ครับ",
  not_confident: "ยังไม่แน่ใจครับ",
  not_durian: "ภาพนี้ไม่ใช่ทุเรียนครับ",
};

interface Prediction {
  class: string;
  confidence: number;
}

interface DiagnosisPayload {
  id: string;
  tier: string;
  top3: Prediction[];
  model_name: string;
  created_at: string;
  disease: {
    name_th: string;
    name_en: string;
    pathogen: string | null;
    slug: string;
    severity: Severity;
    symptoms: string[];
    immediate_actions: string[];
    prevention: string[];
    pending_expert_input: boolean;
  } | null;
}

type Status = "loading" | "ok" | "notfound" | "forbidden" | "error";

declare global {
  interface Window {
    liff?: any;
  }
}

function loadLiffSdk(): Promise<void> {
  if (window.liff) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://static.line-scdn.net/liff/edge/2/sdk.js";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("LIFF SDK failed to load"));
    document.head.appendChild(script);
  });
}

/** Best-effort LIFF login; returns an ID token when running inside LINE. */
async function getIdToken(): Promise<string | null> {
  if (!LIFF_ID) return null;
  try {
    await loadLiffSdk();
    await window.liff.init({ liffId: LIFF_ID });
    if (!window.liff.isLoggedIn()) {
      window.liff.login();
      return null;
    }
    return window.liff.getIDToken();
  } catch {
    // Opened in a plain browser, or the SDK is unreachable. The backend
    // decides whether to allow the request.
    return null;
  }
}

export function ResultView() {
  const params = useSearchParams();
  const diagnosisId = params.get("dg");
  const [status, setStatus] = useState<Status>("loading");
  const [data, setData] = useState<DiagnosisPayload | null>(null);

  useEffect(() => {
    if (!diagnosisId) {
      setStatus("notfound");
      return;
    }
    let cancelled = false;

    (async () => {
      const token = await getIdToken();
      if (cancelled) return;
      try {
        const res = await fetch(
          `${API_BASE}/api/diagnoses/${encodeURIComponent(diagnosisId)}`,
          token ? { headers: { Authorization: `Bearer ${token}` } } : undefined,
        );
        if (cancelled) return;
        if (res.status === 404) return setStatus("notfound");
        if (res.status === 401 || res.status === 403) return setStatus("forbidden");
        if (!res.ok) return setStatus("error");
        setData(await res.json());
        setStatus("ok");
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [diagnosisId]);

  if (status === "loading") return <p>กำลังโหลดผลตรวจ…</p>;

  if (status === "notfound") {
    return (
      <div className="notice">
        <h1>ไม่พบผลตรวจนี้ครับ</h1>
        <p>ลิงก์อาจหมดอายุ หรือถูกลบไปแล้ว ลองส่งรูปใหม่เข้ามาได้เลยครับ</p>
        <Link className="btn btn-primary" href="/diseases/">
          ดูคลังความรู้
        </Link>
      </div>
    );
  }

  if (status === "forbidden") {
    return (
      <div className="notice notice-danger">
        <h1>ดูผลตรวจนี้ไม่ได้ครับ</h1>
        <p>
          ผลตรวจเปิดดูได้เฉพาะเจ้าของรูปเท่านั้น
          กรุณาเปิดลิงก์นี้จากแชท LINE ของคุณเองนะครับ
        </p>
      </div>
    );
  }

  if (status === "error" || !data) {
    return (
      <div className="notice notice-danger">
        <h1>ระบบกำลังยุ่งครับ</h1>
        <p>ลองเปิดใหม่อีกครั้งนะครับ</p>
      </div>
    );
  }

  const disease = data.disease;
  const color = disease ? SEVERITY_COLOR[disease.severity] : "var(--grey-600)";

  return (
    <>
      <h1>{TIER_TITLE[data.tier] ?? "ผลตรวจ"}</h1>

      {disease && (
        <>
          <div style={{ marginBottom: 12 }}>
            <SeverityBadge severity={disease.severity}
                           pending={disease.pending_expert_input} />
          </div>
          <h2>{disease.name_th}</h2>
          <p className="muted">
            {disease.name_en}
            {disease.pathogen ? ` · ${disease.pathogen}` : ""}
          </p>
        </>
      )}

      <div className="card">
        <h3>ผลที่เป็นไปได้</h3>
        <div className="stack">
          {data.top3.map((p, i) => (
            <div key={p.class}>
              <p style={{ marginBottom: 4 }}>
                {i + 1}. {p.class}
              </p>
              <ConfidenceBar
                value={p.confidence}
                color={i === 0 ? color : "var(--grey-600)"}
              />
            </div>
          ))}
        </div>
      </div>

      {disease?.pending_expert_input && (
        <div className="notice">
          <p>
            ข้อมูลคำแนะนำของโรคนี้กำลังจัดทำและรอผู้เชี่ยวชาญตรวจทานอยู่ครับ
            ระหว่างนี้แนะนำให้ปรึกษาเจ้าหน้าที่เกษตรโดยตรงครับ
          </p>
        </div>
      )}

      {disease && !disease.pending_expert_input && (
        <>
          {disease.symptoms.length > 0 && (
            <>
              <h3>🔍 อาการที่พบ</h3>
              <ul className="tick-list">
                {disease.symptoms.slice(0, 3).map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </>
          )}
          {disease.immediate_actions.length > 0 && (
            <>
              <h3>🛠️ วิธีแก้ไขทันที</h3>
              <ul className="tick-list">
                {disease.immediate_actions.slice(0, 3).map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </>
          )}
          <Link className="btn btn-secondary" href={`/diseases/${disease.slug}/`}>
            อ่านข้อมูลโรคนี้แบบเต็ม
          </Link>
        </>
      )}

      <div style={{ marginTop: 24, display: "flex", gap: 12, flexWrap: "wrap" }}>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => {
            if (window.liff?.isInClient?.()) window.liff.closeWindow();
            else window.history.back();
          }}
        >
          ส่งรูปใหม่
        </button>
      </div>

      <p className="muted" style={{ marginTop: 20, fontSize: "var(--text-caption)" }}>
        ตรวจเมื่อ {new Date(data.created_at).toLocaleString("th-TH")} ·
        โมเดล {data.model_name}
      </p>
      <Disclaimer />
    </>
  );
}
