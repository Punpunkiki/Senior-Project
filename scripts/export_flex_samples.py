"""Export every Flex container to docs/flex-samples/ as standalone JSON.

Each file can be pasted straight into the LINE Flex Message Simulator
(https://developers.line.biz/flex-simulator/) to preview the layout without
deploying anything.

    python3 scripts/export_flex_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.flex import builders as B  # noqa: E402
from app.knowledge import Disease, load_knowledge_base  # noqa: E402

# Every class currently has content, so the "awaiting expert content" layout is
# demonstrated with a synthetic entry rather than silently disappearing from
# the samples -- the bot still needs that path when a new class is added.
PENDING_EXAMPLE = Disease(
    class_name="Example_pending", slug="example-pending",
    name_th="โรคตัวอย่างที่ยังไม่มีข้อมูล", name_en="Example pending entry",
    pathogen=None, type=None, severity="watch", pending_expert_input=True,
)

OUT_DIR = REPO_ROOT / "docs" / "flex-samples"
WEB = "https://example.org"
DETAIL = f"{WEB}/result/sample-id"


def main() -> int:
    kb = load_knowledge_base()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        "how-to-photo": B.build_how_to_photo_bubble(),
        "result-confident-disease": B.build_result_bubble(
            kb.by_class("Anthracnose"), 0.91, detail_url=DETAIL),
        "result-severe-disease": B.build_result_bubble(
            kb.by_class("Pink_disease"), 0.87, detail_url=DETAIL),
        "result-healthy": B.build_result_bubble(
            kb.by_class("Healthy"), 0.96, detail_url=DETAIL),
        "result-pending-content": B.build_result_bubble(
            PENDING_EXAMPLE, 0.88, detail_url=DETAIL),
        "result-uncertain-top2": B.build_uncertain_bubble(
            kb.by_class("Anthracnose"), 0.61,
            kb.by_class("Pink_disease"), 0.24, detail_url=DETAIL),
        "result-not-confident": B.build_not_confident_bubble(),
        "result-not-durian": B.build_not_durian_bubble(),
        "model-not-ready": B.build_model_not_ready_bubble(),
        "knowledge-carousel": B.build_knowledge_carousel(
            [d for d in kb.all() if not d.is_healthy], WEB),
    }

    failures = 0
    for name, container in samples.items():
        try:
            B.validate_container(container)
        except ValueError as exc:
            print(f"  [FAIL] {name}: {exc}")
            failures += 1
            continue
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(container, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"  [ok] {name:28s} {B.container_size_bytes(container):>6d} bytes")

    print(f"\n{len(samples) - failures}/{len(samples)} samples written to "
          f"{OUT_DIR.relative_to(REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
