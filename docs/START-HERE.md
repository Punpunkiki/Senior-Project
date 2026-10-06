# เริ่มจากศูนย์ — เปิด VS Code เปล่า ๆ แล้วทำตามนี้

คู่มือนี้พาตั้งแต่ยังไม่มีอะไรในเครื่อง จนบอท LINE ตอบรูปได้
ทำตามทีละข้อ **อย่าข้าม** แต่ละข้อบอกไว้ด้วยว่าทำถูกแล้วจะเห็นอะไร

> **ไฟล์โมเดล `best.pt` อยู่เครื่องคุณที่เดียวพอ ไม่ต้องส่งให้ใคร**
> ไม่ต้องอัปขึ้น GitHub ไม่ต้องส่งให้ Claude ระบบจะอ่านจากเครื่องคุณเอง

---

## ก่อนเริ่ม ต้องมีอะไรบ้าง

ลงสามอย่างนี้ก่อน ถ้ามีแล้วข้ามได้

| โปรแกรม | เช็กว่ามีหรือยัง | โหลดที่ |
|---|---|---|
| Git | `git --version` | [git-scm.com](https://git-scm.com/downloads) |
| Python 3.11 | `python --version` | [python.org](https://www.python.org/downloads/) |
| Node.js 20+ | `node --version` | [nodejs.org](https://nodejs.org/) |

> ตอนลง Python **ติ๊ก "Add Python to PATH"** ด้วย ไม่งั้นคำสั่ง `python` จะไม่เจอ

เปิด VS Code แล้วกด **Terminal → New Terminal** คำสั่งทั้งหมดพิมพ์ในนั้น

---

## ขั้นที่ 1 — โคลนโปรเจกต์

```bash
git clone https://github.com/Punpunkiki/Senior-Project.git
cd Senior-Project
git checkout claude/durian-classifier-handoff-ejwvds
```

เสร็จแล้วใน VS Code กด **File → Open Folder** เลือกโฟลเดอร์ `Senior-Project`

**ถูกต้องเมื่อ:** เห็นโฟลเดอร์ `app`, `web`, `src`, `line-assets` ในแถบซ้าย

---

## ขั้นที่ 2 — สร้างที่อยู่ของ Python แยกไว้

เพื่อไม่ให้ไปปนกับโปรเจกต์อื่นในเครื่อง

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**ถูกต้องเมื่อ:** ข้างหน้าบรรทัดคำสั่งมี `(.venv)` ขึ้นมา

> ทุกครั้งที่เปิด terminal ใหม่ ต้องสั่ง activate ซ้ำ

---

## ขั้นที่ 3 — ลงไลบรารี Python

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
```

บรรทัดแรกโหลดนาน (ประมาณ 200 MB) ใจเย็น ๆ

> **อย่าสั่ง `pip install torch` เฉย ๆ** เพราะจะได้ตัวที่พ่วงไลบรารีการ์ดจอมาอีก 2 GB
> ซึ่งไม่ได้ใช้ และเคยทำให้ import ไม่ขึ้นมาแล้ว

**ถูกต้องเมื่อ:** สั่ง `python -c "import torch; print(torch.__version__)"` แล้วขึ้นเลขเวอร์ชัน

---

## ขั้นที่ 4 — build เว็บ

```bash
cd web
npm install
npm run build
cd ..
```

**ถูกต้องเมื่อ:** ขึ้น `✓ Exporting (2/2)` และมีโฟลเดอร์ `web/out` โผล่มา

> ขั้นนี้ทำครั้งเดียวพอ ยกเว้นแก้โค้ดหน้าเว็บหรือแก้ข้อมูลโรค

---

## ขั้นที่ 5 — เอาไฟล์โมเดลมาวาง

โหลดจาก Google Drive (ที่ Colab เซฟไว้) มาสองไฟล์ แล้ววางให้ตรง path นี้เป๊ะ ๆ

```
Senior-Project/
└── outputs/
    └── convnextv2_tiny/
        ├── best.pt          ← ประมาณ 106 MB
        └── metrics.json     ← ไม่กี่ KB
```

ถ้ายังไม่มีโฟลเดอร์ `convnextv2_tiny` ให้สร้างเอง ลากไฟล์วางใน VS Code ได้เลย

> ชื่อโฟลเดอร์ต้องเป็น `convnextv2_tiny` เป๊ะ ๆ เพราะระบบตั้งไว้ให้ใช้โมเดลตัวนี้
> (เหตุผลอยู่ใน `reports/RESULTS.md` หัวข้อ 3)

---

## ขั้นที่ 6 — สร้างไฟล์ตั้งค่า

```bash
cp .env.example .env
```

**Windows ถ้า `cp` ไม่ได้** ใช้ `copy .env.example .env`

ยังไม่ต้องกรอกอะไร เดี๋ยวค่อยมาเติมตอนได้ค่าจาก LINE

---

## ขั้นที่ 7 — ตรวจว่าทุกอย่างพร้อม

```bash
python scripts/preflight.py
```

ตอนนี้จะเห็นแบบนี้ ซึ่ง**ถูกต้องแล้ว**

```
1. ตั้งค่า (.env)
  ✗ LINE_CHANNEL_SECRET ยังว่าง        ← ปกติ ยังไม่ได้ทำ LINE
2. โมเดล
  ✓ พบไฟล์โมเดล outputs/convnextv2_tiny/best.pt (106.4 MB)
  ✓ จำนวนคลาสตรงกัน (12 คลาส)
  ✓ ทดลองทำนายภาพจริงผ่าน
3. คลังความรู้
  ✓ ครบทุกคลาสที่โมเดลทำนายได้
4. เว็บไซต์
  ✓ build แล้ว (18 หน้า)
```

**สิ่งที่ต้องผ่านตอนนี้คือข้อ 2, 3, 4** ส่วนข้อ 1 ยังแดงได้

**ถ้าข้อ 2 แดง** แปลว่าวางไฟล์ผิดที่ หรือชื่อโฟลเดอร์ไม่ตรง กลับไปขั้นที่ 5

---

## ขั้นที่ 8 — ลองเปิดเว็บดู

```bash
uvicorn app.main:app --reload --port 8000
```

เปิดเบราว์เซอร์ไปที่ **http://localhost:8000**

**ถูกต้องเมื่อ:** เห็นหน้าเว็บหมอทุเรียน มีโลโก้ กดเมนูคลังความรู้ได้ เห็น 10 รายการ

ลองเปิด **http://localhost:8000/health** ต้องขึ้น `"model_loaded": true`

> ถ้าขึ้น `false` แปลว่าโมเดลยังไม่ถูกโหลด กลับไปขั้นที่ 5
>
> กด `Ctrl + C` เพื่อหยุด server

---

## ขั้นที่ 9 — ทำ LINE OA

ยังไม่ต้องปิด terminal อันเดิม **เปิด terminal ใหม่อีกอัน** (กด `+` ใน VS Code)

### 9.1 สร้างบัญชี

ไปที่ [manager.line.biz](https://manager.line.biz) ล็อกอินแล้วกด **สร้างใหม่**
ตั้งชื่อบัญชีว่า `หมอทุเรียน`

### 9.2 เปิด Messaging API

เข้าบัญชีที่เพิ่งสร้าง → **ตั้งค่า → Messaging API → เปิดใช้งาน**
ถ้าถามหา Provider ให้สร้างใหม่ ตั้งชื่ออะไรก็ได้

> ขั้นนี้สำคัญ การสร้างบัญชีเฉย ๆ ยังไม่ได้ token มา

### 9.3 ตั้ง Rich Menu และรูปโปรไฟล์

เปิดไฟล์ `line-assets/AGENT-PROMPT.md` ก๊อปข้อความในกล่องโค้ด
แล้ววางให้ Claude ที่เปิดเบราว์เซอร์ได้ พร้อมแนบไฟล์จากโฟลเดอร์ `line-assets/`

หรือจะทำเองตาม `line-assets/HANDOFF.md` ก็ได้ มีพิกัดปุ่มครบ

### 9.4 เอา token มาใส่

ไปที่ [developers.line.biz](https://developers.line.biz) เข้าช่องที่เพิ่งสร้าง

เปิดไฟล์ `.env` ใน VS Code แล้วเติม

```
LINE_CHANNEL_SECRET=      ← แท็บ Basic settings
LINE_CHANNEL_ACCESS_TOKEN= ← แท็บ Messaging API กดออก token
```

---

## ขั้นที่ 10 — เปิดให้ LINE ยิงเข้ามาได้

LINE ต้องยิงเข้าเครื่องคุณได้ แต่ `localhost` คนนอกเข้าไม่ถึง
ต้องใช้ **ngrok** เปิดทางให้

1. สมัครฟรีที่ [ngrok.com](https://ngrok.com) แล้วโหลดโปรแกรม
2. ทำตามหน้าเว็บเพื่อใส่ authtoken ครั้งเดียว
3. เปิด terminal ใหม่อีกอัน แล้วสั่ง

```bash
ngrok http 8000
```

จะได้ URL หน้าตาแบบ `https://xxxx-xxx-xxx.ngrok-free.app` **ก๊อปเก็บไว้**

> URL นี้เปลี่ยนทุกครั้งที่ปิดแล้วเปิด ngrok ใหม่
> **ตอนนำเสนอ อย่าปิด terminal นี้เด็ดขาด**

---

## ขั้นที่ 11 — ต่อ LINE เข้ากับเครื่อง

```bash
python scripts/preflight.py --url https://xxxx.ngrok-free.app
```

มันจะพิมพ์ค่าที่ต้องเอาไปใส่ให้ครบ ก๊อปวางได้เลย

ที่ [developers.line.biz](https://developers.line.biz) แท็บ **Messaging API**

| ช่อง | ใส่อะไร |
|---|---|
| Webhook URL | `https://xxxx.ngrok-free.app/webhook` แล้วกด **Verify** |
| Use webhook | เปิด |

ที่ [manager.line.biz](https://manager.line.biz) → **ตั้งค่า → การตอบกลับ**

| สวิตช์ | ตั้งเป็น |
|---|---|
| ตอบกลับอัตโนมัติ | **ปิด** |
| Webhook | เปิด |

> ถ้าไม่ปิด auto-reply ผู้ใช้จะได้ข้อความสองชุดซ้อนกันทุกครั้ง

**ถูกต้องเมื่อ:** กด Verify แล้วขึ้น **Success**

---

## ขั้นที่ 12 — ทดสอบจริง

ต้องมีสาม terminal เปิดค้างไว้พร้อมกัน

| Terminal | สั่งอะไร |
|---|---|
| 1 | `uvicorn app.main:app --port 8000` |
| 2 | `ngrok http 8000` |
| 3 | ว่างไว้ เผื่อสั่งอย่างอื่น |

เปิด LINE ในมือถือ แอดเพื่อน OA ของคุณ แล้วไล่ทดสอบ

| ทำอะไร | ควรได้อะไร |
|---|---|
| ทักว่า `คลังความรู้` | การ์ดเลื่อนข้างรวมรายการโรค |
| ทักว่า `วิธีใช้งาน` | การ์ดสอนใช้ + ปุ่มเปิดกล้อง |
| **ส่งรูปใบทุเรียน** | จุดไข่ปลา แล้วได้การ์ดผลตรวจ |
| ส่งรูปอย่างอื่น เช่น รูปแมว | ได้ผลตรวจเหมือนกัน (ดู §ข้อจำกัดด้านล่าง) |
| กด "อ่านรายละเอียดเต็ม" | เปิดหน้าเว็บผลตรวจ |

---

## ข้อจำกัดที่ต้องรู้ก่อนนำเสนอ

**ระบบแยกไม่ออกว่ารูปไม่ใช่ทุเรียน** เพราะตอนเทรนไม่มีรูปในคลาส `not_durian`
ถ้าส่งรูปแมวเข้าไป มันจะตอบว่าเป็นโรคทุเรียนสักอย่าง

ถ้าอาจารย์ถาม ตอบได้ว่านี่เป็นข้อจำกัดที่รู้ตัวและบันทึกไว้แล้วใน
`reports/RESULTS.md` หัวข้อ 8 พร้อมวิธีแก้ (รัน `scripts/prepare_negatives.py`
แล้วเทรนใหม่)

---

## เจอปัญหา

| อาการ | แก้ยังไง |
|---|---|
| `python` ไม่รู้จัก | ตอนลง Python ไม่ได้ติ๊ก Add to PATH ลงใหม่ |
| `(.venv)` หายไป | สั่ง activate ใหม่ (ขั้นที่ 2) |
| `import torch` พัง | ลง torch ผิดแบบ ดูขั้นที่ 3 |
| `/health` ขึ้น `model_loaded: false` | ไฟล์โมเดลวางผิดที่ ดูขั้นที่ 5 |
| กด Verify ไม่ผ่าน | server ไม่ได้เปิด หรือ ngrok ปิดไปแล้ว หรือ URL ไม่ลงท้าย `/webhook` |
| ทักแล้วบอทเงียบ | ดู terminal ที่รัน uvicorn ว่ามี error อะไรขึ้น |
| ทักแล้วตอบข้อความทั่วไป ทั้งที่กดปุ่มเมนู | ข้อความในปุ่ม Rich Menu ไม่ตรงทุกตัวอักษร ดู `line-assets/HANDOFF.md` |
| ได้คำตอบซ้ำสองครั้ง | ยังไม่ได้ปิด auto-reply ดูขั้นที่ 11 |

---

## สรุปคำสั่งที่ใช้บ่อย

```bash
# ทุกครั้งที่เปิด terminal ใหม่
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux

# เปิด server
uvicorn app.main:app --reload --port 8000

# เปิดทางให้ LINE
ngrok http 8000

# ตรวจว่าพร้อมไหม
python scripts/preflight.py

# build เว็บใหม่ (เฉพาะตอนแก้หน้าเว็บหรือข้อมูลโรค)
cd web && npm run build && cd ..
```
