"""app/flex/tokens.py -- "หมอทุเรียน" corporate identity design tokens.

Single source of colour truth. The web front-end mirrors these exact values
as CSS custom properties (see the web app's theme file), so a colour is
changed here and there only -- never inlined into a component or a Flex
builder.
"""
from __future__ import annotations

# --- Brand palette ----------------------------------------------------------
LEAF_700 = "#1F5F3A"  # primary: headers, primary buttons
LEAF_500 = "#2E8B57"  # "ปกติ" (healthy) status, icons
LEAF_100 = "#E6F4EA"  # card backgrounds
DURIAN_500 = "#F2B705"  # accent, highlight, "เฝ้าระวัง" (watch) status
DURIAN_300 = "#FFD95A"  # banners, badges
CREAM_50 = "#FFFBEA"  # main background
HUSK_600 = "#7A8B2E"  # borders, durian-thorn pattern
WARN_500 = "#E8871E"  # "ควรจัดการ" (moderate) status
DANGER_600 = "#C0392B"  # "รุนแรง" (severe) status
INK_900 = "#1E2A1E"  # body text

# Neutrals used for secondary text / rules inside bubbles.
GREY_600 = "#6B7280"
GREY_300 = "#D1D5DB"
WHITE = "#FFFFFF"

# --- Severity -> header colour ---------------------------------------------
# Keys match diseases.json -> "severity".
SEVERITY_COLORS = {
    "healthy": LEAF_500,
    "watch": DURIAN_500,
    "moderate": WARN_500,
    "severe": DANGER_600,
}

# Thai label shown on the severity chip in the result card header.
SEVERITY_LABELS_TH = {
    "healthy": "ปกติ",
    "watch": "เฝ้าระวัง",
    "moderate": "ควรจัดการ",
    "severe": "รุนแรง",
}

# --- Type scale -------------------------------------------------------------
# Farmers are often older and on small phones: body text never goes below
# "md" in a Flex bubble, and headings are "xl"/"xxl".
SIZE_TITLE = "xxl"
SIZE_SUBTITLE = "sm"
SIZE_HEADING = "lg"
SIZE_BODY = "md"
SIZE_CAPTION = "xs"

CORNER_RADIUS = "16px"


def severity_color(severity: str) -> str:
    """Header colour for a severity level, defaulting to the 'watch' amber for
    anything unrecognised so a bad KB value can never render as 'healthy'."""
    return SEVERITY_COLORS.get(severity, DURIAN_500)


def severity_label_th(severity: str) -> str:
    return SEVERITY_LABELS_TH.get(severity, "เฝ้าระวัง")
