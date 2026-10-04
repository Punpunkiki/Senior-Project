"""Check everything the bot needs BEFORE touching the LINE console.

Debugging a failed webhook verification from the LINE console gives you
almost no information, so this answers the question "is my side actually
ready?" first, and prints the exact values to paste into LINE.

    python3 scripts/preflight.py
    python3 scripts/preflight.py --url https://abc123.ngrok-free.app

Exit code is 0 when nothing is broken, 1 when something is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OK, WARN, FAIL = "ok", "warn", "FAIL"
_SYM = {OK: "  \033[32m✓\033[0m", WARN: "  \033[33m!\033[0m", FAIL: "  \033[31m✗\033[0m"}

results: list[tuple[str, str]] = []


def say(level: str, msg: str, hint: str = "") -> None:
    results.append((level, msg))
    print(f"{_SYM[level]} {msg}")
    if hint:
        for line in hint.splitlines():
            print(f"      {line}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# --------------------------------------------------------------------------- #
def check_env():
    section("1. ตั้งค่า (.env)")
    if not (REPO / ".env").exists():
        say(FAIL, "ไม่พบไฟล์ .env",
            "cp .env.example .env  แล้วกรอกค่าจาก LINE Developers")
        return None

    from app.config import settings

    required = {
        "LINE_CHANNEL_SECRET": settings.line_channel_secret,
        "LINE_CHANNEL_ACCESS_TOKEN": settings.line_channel_access_token,
    }
    for name, value in required.items():
        if value:
            say(OK, f"{name} ตั้งแล้ว ({len(value)} ตัวอักษร)")
        else:
            say(FAIL, f"{name} ยังว่าง", "ดู LINE Developers > ช่อง Messaging API")

    if settings.liff_id:
        say(OK, f"LIFF_ID = {settings.liff_id}")
    else:
        say(WARN, "LIFF_ID ยังว่าง — ปุ่ม 'อ่านรายละเอียดเต็ม' จะชี้ไปเว็บแทน LIFF")

    if settings.liff_channel_id:
        say(OK, "LIFF_CHANNEL_ID ตั้งแล้ว — หน้าผลตรวจจะตรวจสอบเจ้าของได้")
    else:
        say(WARN, "LIFF_CHANNEL_ID ยังว่าง",
            "หน้าผลตรวจจะเปิดได้โดยไม่ตรวจตัวตน ใครมีลิงก์ก็ดูได้")
    return settings


def check_model(settings):
    section("2. โมเดล")
    if settings is None:
        say(FAIL, "ข้ามเพราะยังไม่มี .env")
        return

    from src.utils import load_config

    cfg = load_config(settings.config_yaml_path)
    names = [m["name"] for m in cfg["models"]]
    if settings.model_name not in names:
        say(FAIL, f"MODEL_NAME '{settings.model_name}' ไม่มีใน config.yaml",
            f"ชื่อที่ใช้ได้: {', '.join(names)}")
        return
    say(OK, f"MODEL_NAME = {settings.model_name}")

    out_dir = Path(cfg["paths"]["outputs_dir"]) / settings.model_name
    ckpt = out_dir / "best.pt"
    if not ckpt.exists():
        say(FAIL, f"ไม่พบไฟล์โมเดลที่ {ckpt}",
            "โหลด best.pt จาก Colab มาวางตรงนี้\n"
            "บอทจะตอบ 'ระบบยังไม่พร้อมใช้งาน' ทุกรูปจนกว่าจะมีไฟล์นี้")
        return

    size_mb = ckpt.stat().st_size / 1024 / 1024
    say(OK, f"พบไฟล์โมเดล {ckpt} ({size_mb:.1f} MB)")

    # Load it for real: a checkpoint trained with a different class count
    # fails here instead of at the first farmer's photo.
    try:
        import torch

        state = torch.load(ckpt, map_location="cpu")
        sd = state.get("model", state)
        head = [v for k, v in sd.items() if k.endswith(("classifier.bias",
                                                        "head.fc.bias",
                                                        "fc.bias", "head.bias"))]
        n_cfg = cfg["data"]["num_classes"]
        if head:
            n_ckpt = head[-1].shape[0]
            if n_ckpt == n_cfg:
                say(OK, f"จำนวนคลาสตรงกัน ({n_ckpt} คลาส)")
            else:
                say(FAIL, f"จำนวนคลาสไม่ตรง: ไฟล์โมเดลมี {n_ckpt} "
                          f"แต่ config.yaml ระบุ {n_cfg}",
                    "เทรนด้วย config คนละชุด ต้องเทรนใหม่หรือแก้ config")
                return
        if "val_macro_f1" in state:
            say(OK, f"val macro-F1 ตอนเทรน = {state['val_macro_f1']:.4f}")
    except Exception as exc:
        say(FAIL, f"เปิดไฟล์โมเดลไม่ได้: {exc}")
        return

    metrics = out_dir / "metrics.json"
    if metrics.exists():
        temp = json.loads(metrics.read_text()).get("temperature")
        say(OK, f"พบ metrics.json (temperature = {temp})")
    else:
        say(WARN, "ไม่พบ metrics.json — ความมั่นใจจะไม่ได้ปรับเทียบ",
            f"รัน: python -m src.evaluate --model {settings.model_name}")

    # Real inference on a synthetic image proves the whole path works.
    try:
        from PIL import Image

        from app.ml.predictor import DurianPredictor

        pred = DurianPredictor(settings.config_yaml_path, settings.model_name)
        if not pred.ready:
            say(FAIL, "predictor ไม่พร้อม ทั้งที่มีไฟล์โมเดล")
            return
        top3 = pred.predict(Image.new("RGB", (400, 400), (120, 150, 90)), top_k=3)
        say(OK, "ทดลองทำนายภาพจริงผ่าน: "
                + ", ".join(f"{s.label} {s.confidence:.0%}" for s in top3))
    except Exception as exc:
        say(FAIL, f"ทำนายไม่สำเร็จ: {exc}")


def check_knowledge(settings):
    section("3. คลังความรู้")
    from app.knowledge import load_knowledge_base
    from src.utils import load_config

    kb = load_knowledge_base()
    cfg = load_config(settings.config_yaml_path if settings
                      else REPO / "config.yaml")
    missing = kb.missing_for(cfg["data"]["classes"])
    if missing:
        say(FAIL, f"ไม่มีข้อมูลสำหรับคลาส: {', '.join(missing)}",
            "บอทจะตอบ 'ยังไม่แน่ใจ' เมื่อทำนายได้คลาสเหล่านี้")
    else:
        say(OK, f"ครบทุกคลาสที่โมเดลทำนายได้ ({len(kb.all())} รายการ)")

    pending = [d.name_th for d in kb.pending()]
    if pending:
        say(WARN, f"{len(pending)} รายการยังรอเนื้อหา: {', '.join(pending)}")
    unreviewed = [d for d in kb.all() if not d.reviewed_by_expert]
    if unreviewed:
        say(WARN, f"{len(unreviewed)} รายการยังไม่ผ่านการตรวจทานของผู้เชี่ยวชาญ",
            "ดู docs/knowledge-review.md")


def check_website():
    section("4. เว็บไซต์")
    index = REPO / "web" / "out" / "index.html"
    if index.exists():
        pages = len(list((REPO / "web" / "out").rglob("index.html")))
        say(OK, f"build แล้ว ({pages} หน้า)")
    else:
        say(FAIL, "ยังไม่ได้ build เว็บ",
            "cd web && npm ci && npm run build")


def check_line_token(settings):
    section("5. ติดต่อ LINE จริง")
    if settings is None or not settings.line_channel_access_token:
        say(WARN, "ข้าม เพราะยังไม่มี access token")
        return
    try:
        import httpx

        r = httpx.get("https://api.line.me/v2/bot/info",
                      headers={"Authorization":
                               f"Bearer {settings.line_channel_access_token}"},
                      timeout=10)
        if r.status_code == 200:
            info = r.json()
            say(OK, f"token ใช้ได้ — บัญชี: {info.get('displayName')} "
                    f"({info.get('basicId')})")
        elif r.status_code == 401:
            say(FAIL, "token ไม่ถูกต้องหรือหมดอายุ",
                "ออก token ใหม่ที่ LINE Developers > Messaging API")
        else:
            say(WARN, f"LINE ตอบ {r.status_code}")
    except Exception as exc:
        say(WARN, f"ต่อ LINE ไม่ได้ (อาจเป็นเรื่องเน็ต): {exc}")


def print_next_steps(settings, public_url: str | None):
    section("6. ค่าที่ต้องเอาไปใส่ใน LINE Developers")
    base = public_url or (settings.web_base_url if settings else "")
    if not base or "localhost" in base:
        print("      ยังไม่รู้ URL สาธารณะ — รัน ngrok แล้วสั่งใหม่พร้อม --url")
        print("      ตัวอย่าง: python3 scripts/preflight.py --url https://xxx.ngrok-free.app")
        return
    base = base.rstrip("/")
    print(f"      Webhook URL      {base}/webhook")
    print(f"      LIFF Endpoint    {base}/result")
    print(f"      Rich Menu ปุ่ม A  {base}")
    print(f"      WEB_BASE_URL     {base}   (ใส่ใน .env ด้วย)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="URL สาธารณะ เช่นที่ได้จาก ngrok")
    args = ap.parse_args()

    os.chdir(REPO)
    print("\033[1mตรวจความพร้อมก่อนต่อ LINE\033[0m")

    settings = check_env()
    check_model(settings)
    check_knowledge(settings)
    check_website()
    check_line_token(settings)
    print_next_steps(settings, args.url)

    fails = sum(1 for lvl, _ in results if lvl == FAIL)
    warns = sum(1 for lvl, _ in results if lvl == WARN)
    print()
    if fails:
        print(f"\033[31m{fails} อย่างต้องแก้ก่อน\033[0m"
              + (f" (และ {warns} ข้อควรระวัง)" if warns else ""))
        return 1
    print(f"\033[32mพร้อมต่อ LINE ได้\033[0m"
          + (f" (มี {warns} ข้อควรระวัง)" if warns else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
