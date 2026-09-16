"""app/messages.py -- all farmer-facing Thai copy in one place.

Tone: a kind village plant doctor -- warm, direct, no unexplained jargon,
consistently ending with "ครับ". Short lines, because most users read this
on a small phone in the sun.

The KEYWORD_* values are a contract with the Rich Menu configured in LINE OA
Manager: each menu button sends one of these exact strings, and
app/line_handler.py routes on them. Changing one means changing the Rich
Menu too (see line-assets/HANDOFF.md).
"""
from __future__ import annotations

# --- Rich Menu keywords (must match the menu buttons byte-for-byte) -------
KEYWORD_DIAGNOSE = "ตรวจโรคทุเรียน"
KEYWORD_HOW_TO_PHOTO = "วิธีถ่ายรูป"
KEYWORD_KNOWLEDGE = "คลังความรู้"
KEYWORD_CONTACT = "ติดต่อเจ้าหน้าที่"

# --- Shared disclaimer (must appear on every diagnosis result) -----------
DISCLAIMER = (
    "ผลนี้เป็นการประเมินเบื้องต้นจาก AI "
    "ควรปรึกษาเจ้าหน้าที่เกษตรก่อนใช้สารเคมี "
    "และใช้สารตามอัตราบนฉลากเสมอ"
)

# --- Photo guidance -----------------------------------------------------------
HOW_TO_PHOTO_TITLE = "วิธีถ่ายรูปให้ตรวจแม่น"
HOW_TO_PHOTO_TIPS = [
    "ถ่ายใกล้ ๆ ให้เห็นรอยโรคชัด ๆ ครับ",
    "ถ่ายกลางแจ้งหรือที่มีแสงธรรมชาติ อย่าให้เงาบัง",
    "ถ่ายทีละจุด อย่าให้ภาพเบลอ",
]
HOW_TO_PHOTO_FOOTER = "พร้อมแล้วกดปุ่มด้านล่างเพื่อส่งรูปได้เลยครับ"

SEND_PHOTO_PROMPT = "ส่งรูปใบ กิ่ง ลำต้น หรือผลที่สงสัยมาได้เลยครับ"

# --- Result copy ----------------------------------------------------------------
RESULT_CONFIDENT_PREFIX = "ตรวจพบ"
RESULT_UNCERTAIN_TITLE = "อาจเป็นโรคนี้ครับ"
RESULT_UNCERTAIN_HINT = (
    "หมอยังไม่ค่อยแน่ใจ ลองถ่ายเพิ่มอีกมุมหนึ่งแล้วส่งมาใหม่ "
    "จะช่วยให้ตรวจแม่นขึ้นครับ"
)
RESULT_HEALTHY_TITLE = "ต้นดูแข็งแรงดีครับ"
RESULT_HEALTHY_BODY = "ไม่พบร่องรอยของโรคในรูปนี้ ดูแลแบบนี้ต่อไปได้เลยครับ"

RESULT_NOT_CONFIDENT_TITLE = "ยังไม่แน่ใจครับ"
RESULT_NOT_CONFIDENT_BODY = (
    "หมอดูรูปนี้แล้วยังบอกไม่ได้ชัด ๆ ครับ "
    "ลองถ่ายใหม่ให้ใกล้ขึ้นและมีแสงมากกว่านี้ "
    "หรือกดปรึกษาเจ้าหน้าที่ได้เลยครับ"
)

RESULT_NOT_DURIAN_TITLE = "ภาพนี้ไม่ใช่ทุเรียนครับ"
RESULT_NOT_DURIAN_BODY = (
    "หมอตรวจได้เฉพาะรูปต้นทุเรียนครับ "
    "ลองถ่ายใบ กิ่ง ลำต้น หรือผลทุเรียนที่สงสัยแล้วส่งมาใหม่นะครับ"
)

PENDING_DISEASE_NOTICE = (
    "ข้อมูลคำแนะนำของโรคนี้กำลังจัดทำและรอผู้เชี่ยวชาญตรวจทานอยู่ครับ "
    "ระหว่างนี้แนะนำให้ปรึกษาเจ้าหน้าที่เกษตรโดยตรงครับ"
)

# --- Section headings inside the result card ----------------------------------
HEADING_SYMPTOMS = "🔍 อาการที่พบ"
HEADING_CAUSES = "🦠 สาเหตุ"
HEADING_ACTIONS = "🛠️ วิธีแก้ไขทันที"
HEADING_PREVENTION = "🛡️ วิธีป้องกัน"
HEADING_CARE_TIPS = "🌱 เคล็ดลับดูแลต้น"

# --- Buttons ------------------------------------------------------------------
BTN_FULL_DETAIL = "อ่านรายละเอียดเต็ม"
BTN_NEW_PHOTO = "ตรวจรูปใหม่"
BTN_CONTACT_STAFF = "ปรึกษาเจ้าหน้าที่"
BTN_TAKE_PHOTO = "📷 ถ่ายรูป"
BTN_PICK_PHOTO = "🖼️ เลือกรูป"

# --- Knowledge library ----------------------------------------------------------
KNOWLEDGE_TITLE = "คลังความรู้โรคทุเรียน"
KNOWLEDGE_SUBTITLE = "เลือกดูรายละเอียดแต่ละโรคได้เลยครับ"
BTN_READ_DISEASE = "อ่านรายละเอียด"

# --- Fallbacks / errors ---------------------------------------------------------
FALLBACK_TEXT = (
    "สวัสดีครับ กดเมนูด้านล่าง หรือส่งรูปที่สงสัยเข้ามาให้หมอดูได้เลยครับ"
)
ERROR_BUSY = "ระบบกำลังยุ่ง ลองส่งใหม่อีกครั้งนะครับ"
ERROR_MODEL_NOT_READY = (
    "ตอนนี้ระบบตรวจโรคยังไม่พร้อมใช้งานครับ "
    "ขออภัยด้วย ลองใหม่อีกครั้งภายหลัง หรือปรึกษาเจ้าหน้าที่ได้เลยครับ"
)
ERROR_IMAGE_TOO_LARGE = (
    "รูปนี้ใหญ่เกินไปครับ ลองถ่ายใหม่หรือส่งรูปที่เล็กลงนะครับ"
)
