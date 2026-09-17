"use client";

import { useMemo, useState } from "react";
import { Disease, PART_FILTERS, TYPE_LABEL_TH } from "@/lib/types";
import { DiseaseCard } from "@/components/ui";

const TYPE_FILTERS = ["fungus", "insect", "algae", "oomycete"];

export function DiseaseList({ diseases }: { diseases: Disease[] }) {
  const [query, setQuery] = useState("");
  const [part, setPart] = useState<string | null>(null);
  const [type, setType] = useState<string | null>(null);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return diseases.filter((d) => {
      const matchesQuery =
        !q ||
        d.name_th.toLowerCase().includes(q) ||
        d.name_en.toLowerCase().includes(q) ||
        (d.pathogen ?? "").toLowerCase().includes(q);
      const matchesPart = !part || d.affected_parts.includes(part);
      const matchesType = !type || d.type === type;
      return matchesQuery && matchesPart && matchesType;
    });
  }, [diseases, query, part, type]);

  return (
    <>
      <div className="filter-bar">
        <label className="visually-hidden" htmlFor="disease-search">
          ค้นหาชื่อโรค
        </label>
        <input
          id="disease-search"
          type="search"
          placeholder="ค้นหาชื่อโรค เช่น ราสีชมพู"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <div className="filter-bar" role="group" aria-label="กรองตามชนิดสาเหตุ">
        {TYPE_FILTERS.map((t) => (
          <button
            key={t}
            type="button"
            className="chip"
            aria-pressed={type === t}
            onClick={() => setType(type === t ? null : t)}
          >
            {TYPE_LABEL_TH[t] ?? t}
          </button>
        ))}
      </div>

      <div className="filter-bar" role="group" aria-label="กรองตามส่วนของต้น">
        {PART_FILTERS.map((p) => (
          <button
            key={p}
            type="button"
            className="chip"
            aria-pressed={part === p}
            onClick={() => setPart(part === p ? null : p)}
          >
            {p}
          </button>
        ))}
      </div>

      <p className="muted" aria-live="polite">
        พบ {visible.length} รายการ
      </p>

      {visible.length === 0 ? (
        <div className="notice">
          <p>ไม่พบโรคที่ตรงกับที่ค้นหาครับ ลองพิมพ์คำอื่นหรือล้างตัวกรองดูนะครับ</p>
        </div>
      ) : (
        <div className="card-grid">
          {visible.map((d) => (
            <DiseaseCard key={d.class_name} disease={d} />
          ))}
        </div>
      )}
    </>
  );
}
