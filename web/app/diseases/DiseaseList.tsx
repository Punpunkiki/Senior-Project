"use client";

import { useMemo, useState } from "react";
import { Disease } from "@/lib/types";
import { DiseaseCard } from "@/components/ui";

/**
 * One long list, no category chips. Farmers arriving from the LINE card are
 * looking for one specific thing they just saw on their tree, so a search box
 * over a single scrollable column beats a filter taxonomy they would have to
 * learn first.
 */
export function DiseaseList({ diseases }: { diseases: Disease[] }) {
  const [query, setQuery] = useState("");

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return diseases;
    return diseases.filter((d) =>
      [d.name_th, d.name_en, d.pathogen ?? "", ...d.symptoms,
       ...d.affected_parts]
        .join(" ")
        .toLowerCase()
        .includes(q),
    );
  }, [diseases, query]);

  return (
    <>
      <div className="filter-bar">
        <label className="visually-hidden" htmlFor="disease-search">
          ค้นหาสิ่งผิดปกติ
        </label>
        <input
          id="disease-search"
          type="search"
          placeholder="ค้นหา เช่น ยางไหล ใบเหลือง เพลี้ย"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <p className="muted" aria-live="polite">
        พบ {visible.length} รายการ
      </p>

      {visible.length === 0 ? (
        <div className="notice">
          <p>ไม่พบรายการที่ตรงกับที่ค้นหา ลองพิมพ์คำอื่นดู</p>
        </div>
      ) : (
        <div className="card-list">
          {visible.map((d) => (
            <DiseaseCard key={d.class_name} disease={d} />
          ))}
        </div>
      )}
    </>
  );
}
