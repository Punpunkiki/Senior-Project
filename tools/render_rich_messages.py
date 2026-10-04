"""Render LINE Rich messages (1040x1040) from the project's CI.

Design is deliberately not invented here: the colours, the olive panel, the
Thai-headline-over-letterspaced-English pattern and the yellow "กดตรงนี้" pill
are lifted from the Rich Menu artwork the designer produced, so a broadcast
and the menu read as the same product. Colours were sampled from that file
and matched the CI tokens already in app/flex/tokens.py.

Fonts are embedded as base64 so the render does not depend on network access
at build time, and so the output uses the same Kanit/Sarabun the website does.

    python3 tools/render_rich_messages.py            # writes line-assets/
    python3 tools/render_rich_messages.py --check    # verify output only

Requires: playwright (plus its chromium), and the TTFs under tools/fonts/.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FONT_DIR = Path(__file__).resolve().parent / "fonts"
OUT_DIR = REPO / "line-assets"
LOGO = REPO / "line-assets" / "logo-transparent-1024.png"

SIZE = 1040                 # LINE Rich message square
MAX_BYTES = 10 * 1024 * 1024

# --- CI, sampled from the Rich Menu artwork ------------------------------
GREEN = "#7A8B2E"           # husk-600, the panel ground
GREEN_DEEP = "#5E6C21"      # shade for the thorn motif
CREAM_ICON = "#FFF093"      # the pale line-icon yellow
DURIAN = "#F2B705"          # pill fill
PILL_INK = "#452615"        # pill text + arrow disc
WHITE = "#FFFFFF"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def font_face(name: str, file: str, weight: int) -> str:
    return (f"@font-face{{font-family:'{name}';font-weight:{weight};"
            f"font-style:normal;src:url(data:font/ttf;base64,"
            f"{_b64(FONT_DIR / file)}) format('truetype');}}")


# --- line icons, drawn in the same stroke style as the menu's icons ------
ICONS = {
    "camera": """
      <rect x="14" y="30" width="92" height="62" rx="12"/>
      <path d="M40 30l8-12h24l8 12"/>
      <circle cx="60" cy="61" r="19"/>
      <circle cx="90" cy="45" r="3.5" fill="currentColor" stroke="none"/>""",
    "rain": """
      <path d="M34 54a20 20 0 0 1 38-9 15 15 0 0 1 20 9 14 14 0 0 1-2 28H38a16 16 0 0 1-4-28z"/>
      <path d="M40 92l-6 16M60 92l-6 16M80 92l-6 16"/>""",
    "sprout": """
      <path d="M60 100V54"/>
      <path d="M60 60C60 42 46 30 28 30c0 18 14 30 32 30z"/>
      <path d="M60 68c0-15 12-25 27-25 0 15-12 25-27 25z"/>
      <path d="M44 100h32"/>""",
}


def icon_svg(name: str) -> str:
    return (f'<svg viewBox="0 0 120 120" fill="none" stroke="currentColor" '
            f'stroke-width="7" stroke-linecap="round" stroke-linejoin="round">'
            f'{ICONS[name]}</svg>')


MESSAGES = [
    {
        "slug": "richmsg-welcome",
        "th": "ส่งรูปมาให้หมอดูได้เลย",
        "en": "HOW TO START",
        "lines": ["ถ่ายใบ กิ่ง ลำต้น หรือผลที่ดูผิดปกติ",
                  "ส่งเข้าแชทนี้ ไม่ต้องกดเมนูอะไรก่อน"],
        "icon": "camera",
        "cta": "เริ่มใช้งาน",
    },
    {
        "slug": "richmsg-rainy-alert",
        "th": "หน้าฝนระวังรากเน่าโคนเน่า",
        "en": "RAINY SEASON ALERT",
        "lines": ["ฝนชุกดินแฉะ เชื้อเข้าโคนต้นได้ง่าย",
                  "เดินดูโคนต้น ถ้าเจอยางไหลรีบจัดการ"],
        "icon": "rain",
        "cta": "ดูวิธีจัดการ",
    },
    {
        "slug": "richmsg-flush-alert",
        "th": "ช่วงแตกใบอ่อนระวังเพลี้ยไฟ",
        "en": "YOUNG LEAF ALERT",
        "lines": ["หน้าแล้งเพลี้ยไฟระบาดหนัก",
                  "ใบอ่อนหงิกงอ ดอกร่วง ผลบิดเบี้ยว"],
        "icon": "sprout",
        "cta": "ดูวิธีจัดการ",
    },
]


def build_html(msg: dict) -> str:
    fonts = (font_face("Kanit", "Kanit-SemiBold.ttf", 600)
             + font_face("Kanit", "Kanit-Regular.ttf", 400)
             + font_face("Sarabun", "Sarabun-Regular.ttf", 400))
    body_lines = "".join(f"<p>{ln}</p>" for ln in msg["lines"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{fonts}
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:{SIZE}px;height:{SIZE}px}}
body{{
  background:{GREEN};
  /* the durian-thorn motif from the site, barely there: the menu's
     green panel is flat, so this must not compete with it */
  background-image:
    linear-gradient(135deg,{GREEN_DEEP}1F 25%,transparent 25%),
    linear-gradient(225deg,{GREEN_DEEP}1F 25%,transparent 25%);
  background-size:96px 96px;
  display:flex;flex-direction:column;align-items:center;
  justify-content:space-between;
  padding:74px 64px 64px;
  font-family:'Sarabun',sans-serif;color:{WHITE};text-align:center;
}}
h1{{font-family:'Kanit',sans-serif;font-weight:600;font-size:76px;
   line-height:1.18;letter-spacing:-.5px;
   text-shadow:0 3px 0 rgba(0,0,0,.18)}}
.en{{font-family:'Kanit',sans-serif;font-weight:400;font-size:27px;
   letter-spacing:.34em;color:{CREAM_ICON};margin-top:16px}}
.icon{{color:{CREAM_ICON};width:300px;height:300px;opacity:.95}}
.icon svg{{width:100%;height:100%}}
.lines{{font-size:33px;line-height:1.5;opacity:.97}}
.pill{{
  display:flex;align-items:center;gap:22px;
  background:{DURIAN};border-radius:999px;
  padding:16px 30px 16px 18px;
  box-shadow:0 7px 0 rgba(0,0,0,.22);
}}
.pill img{{width:92px;height:92px;object-fit:contain}}
.pill .label{{font-family:'Kanit',sans-serif;font-weight:600;font-size:44px;
  color:{PILL_INK}}}
.disc{{width:60px;height:60px;border-radius:50%;background:{PILL_INK};
  display:flex;align-items:center;justify-content:center}}
.disc svg{{width:26px;height:26px}}
</style></head><body>
  <div>
    <h1>{msg['th']}</h1>
    <div class="en">{msg['en']}</div>
  </div>
  <div class="icon">{icon_svg(msg['icon'])}</div>
  <div class="lines">{body_lines}</div>
  <div class="pill">
    <img src="data:image/png;base64,{_b64(LOGO)}" alt="">
    <span class="label">{msg['cta']}</span>
    <span class="disc"><svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"
      fill="{DURIAN}"/></svg></span>
  </div>
</body></html>"""


async def render_all(check_only: bool = False) -> int:
    from playwright.async_api import async_playwright

    missing = [f.name for f in
               (FONT_DIR / "Kanit-SemiBold.ttf", FONT_DIR / "Kanit-Regular.ttf",
                FONT_DIR / "Sarabun-Regular.ttf") if not f.exists()]
    if missing:
        print(f"ERROR: missing fonts in {FONT_DIR}: {missing}")
        return 1
    if not LOGO.exists():
        print(f"ERROR: missing {LOGO}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/_richmsg.html")
    failures = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/opt/pw-browsers/chromium")
        page = await browser.new_page(
            viewport={"width": SIZE, "height": SIZE}, device_scale_factor=1)
        for msg in MESSAGES:
            tmp.write_text(build_html(msg), encoding="utf-8")
            await page.goto(f"file://{tmp}", wait_until="load")
            await page.wait_for_timeout(120)   # let the embedded fonts settle
            out = OUT_DIR / f"{msg['slug']}-{SIZE}x{SIZE}.jpg"
            if check_only:
                print(f"  [dry-run] would write {out.name}")
                continue
            await page.screenshot(path=str(out), type="jpeg", quality=92)
            size = out.stat().st_size
            ok = size <= MAX_BYTES
            failures += 0 if ok else 1
            print(f"  [{'ok' if ok else 'FAIL'}] {out.name:38s} "
                  f"{size/1024:6.0f} KB")
        await browser.close()
    tmp.unlink(missing_ok=True)

    if not check_only:
        print(f"\n{len(MESSAGES) - failures}/{len(MESSAGES)} rich messages "
              f"-> {OUT_DIR.relative_to(REPO)}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="build the pages but do not write images")
    args = ap.parse_args()
    return asyncio.run(render_all(args.check))


if __name__ == "__main__":
    sys.exit(main())
