# คู่มือติดตั้งและ deploy — หมอทุเรียน

เอกสารนี้ครอบคลุมสามวิธี เรียงจากง่ายไปยาก เลือกใช้ตามสถานการณ์

| วิธี | เหมาะกับ | ค่าใช้จ่าย |
|---|---|---|
| [1. รัน local + ngrok](#1-รัน-local--ngrok) | **ตอนนำเสนอ** และตอนพัฒนา | ฟรี |
| [2. Docker บนเครื่องตัวเอง](#2-docker-บนเครื่องตัวเอง) | ทดสอบว่า image ใช้ได้ก่อนขึ้นจริง | ฟรี |
| [3. Google Cloud Run](#3-google-cloud-run) | ให้อาจารย์กดเปิดดูเองได้ตลอด | ฟรี (free tier) |

> **แนะนำสำหรับการนำเสนอ: วิธีที่ 1** ไม่มี cold start ไม่มีค่าใช้จ่าย
> ควบคุมได้เต็มที่ ถ้าเน็ตห้องนำเสนอมีปัญหาก็ยังเห็นเว็บทำงานได้

---

## สิ่งที่ต้องมีก่อน

### ก. ไฟล์โมเดลที่เทรนแล้ว

วางไว้ที่ `outputs/<ชื่อโมเดล>/best.pt` เช่น `outputs/efficientnet_b0/best.pt`

**ถ้ายังไม่มีไฟล์นี้ ระบบยังรันได้ปกติ** แต่จะตอบว่า "ระบบยังไม่พร้อมใช้งาน"
ทุกครั้งที่มีคนส่งรูป และจะขึ้น log ตัวแดงตอนเริ่มทำงาน ซึ่งตั้งใจให้เป็นแบบนั้น
เพื่อไม่ให้เดาคำตอบจากโมเดลที่ยังไม่ได้เทรน

ชื่อโมเดลที่จะใช้ตั้งที่ `MODEL_NAME` ใน `.env` ต้องตรงกับชื่อใน `config.yaml`
ส่วน `models:` (`convnextv2_tiny` / `swin_tiny` / `efficientnet_b0`)

หลังเทรนเสร็จแนะนำให้รัน `python -m src.evaluate --model <ชื่อ>` ด้วย
เพื่อให้ได้ `metrics.json` ซึ่งมีค่า temperature สำหรับปรับความมั่นใจให้ตรงความจริง
ถ้าไม่มีไฟล์นี้ระบบจะใช้ค่าเริ่มต้นและเตือนใน log

### ข. บัญชี LINE

1. สมัคร [LINE Developers](https://developers.line.biz/) แล้วสร้าง **Messaging API channel**
2. จดค่าสองอันนี้ไว้
   - **Channel secret** (แท็บ Basic settings)
   - **Channel access token** (แท็บ Messaging API กดออก token ยาว)
3. สร้าง **LIFF app** ในแท็บ LIFF
   - Size: `Full`
   - Endpoint URL: `https://<โดเมนของคุณ>/result`
   - Scope: ต้องติ๊ก `profile` และ `openid`
   - จด **LIFF ID** กับ **Channel ID** ของช่องที่สร้าง LIFF ไว้

### ค. ไฟล์ `.env`

```bash
cp .env.example .env
```

แล้วกรอกค่า อย่างน้อยต้องมี

```
LINE_CHANNEL_SECRET=xxxxx
LINE_CHANNEL_ACCESS_TOKEN=xxxxx
LIFF_ID=1234567890-abcdefgh
LIFF_CHANNEL_ID=1234567890
WEB_BASE_URL=https://<โดเมนของคุณ>
```

> ⚠️ **ถ้าไม่ตั้ง `LIFF_CHANNEL_ID`** หน้าผลตรวจจะตรวจสอบตัวตนไม่ได้
> แปลว่าใครที่ได้ลิงก์ไปก็เปิดดูผลตรวจของคนอื่นได้ ระบบจะเตือนตอนเริ่มทำงาน
> แต่ยังทำงานต่อ **ต้องตั้งก่อนใช้งานจริง**

---

## 1. รัน local + ngrok

### ติดตั้ง

```bash
# ฝั่ง Python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt

# ฝั่งเว็บ
cd web && npm ci && npm run build && cd ..
```

> ติดตั้ง torch จาก `--index-url .../whl/cpu` ตามนี้เสมอ อย่าใช้
> `pip install torch` เฉย ๆ เพราะจะได้ตัวที่พ่วง CUDA มาด้วยประมาณ 2 GB
> และเคยทำให้ import ไม่ขึ้นเพราะไลบรารี CUDA คนละรุ่นปนกัน

### รัน

```bash
uvicorn app.main:app --reload --port 8000
```

เปิด http://localhost:8000 จะเห็นเว็บ และ http://localhost:8000/health
จะบอกว่าโหลดโมเดลได้หรือยัง

### เปิดให้ LINE ยิงเข้ามาได้

```bash
ngrok http 8000
```

ngrok จะให้ URL หน้าตาแบบ `https://xxxx.ngrok-free.app` เอาไปตั้งที่

- LINE Developers → Messaging API → **Webhook URL** = `https://xxxx.ngrok-free.app/webhook`
- กด **Verify** ต้องขึ้น Success
- เปิด **Use webhook** และ**ปิด Auto-reply messages**
- LIFF → Endpoint URL = `https://xxxx.ngrok-free.app/result`
- `.env` → `WEB_BASE_URL=https://xxxx.ngrok-free.app`

> URL ของ ngrok เปลี่ยนทุกครั้งที่รันใหม่ (เว้นแต่ใช้แบบเสียเงิน)
> ต้องมาแก้ที่ LINE ใหม่ทุกรอบ **ตั้งก่อนนำเสนอสัก 10 นาที แล้วอย่าปิด terminal**

---

## 2. Docker บนเครื่องตัวเอง

```bash
docker build -t mor-durian .
docker run --rm -p 8080:8080 --env-file .env mor-durian
```

เปิด http://localhost:8080

### ข้อมูลหายเมื่อปิด container

ค่าเริ่มต้นเก็บฐานข้อมูลกับรูปไว้ใน `/tmp` ซึ่งหายเมื่อ container ดับ
ถ้าอยากให้อยู่ถาวร

```bash
docker run --rm -p 8080:8080 --env-file .env \
  -v "$(pwd)/runtime:/data" \
  -e DATABASE_URL="sqlite:////data/durian.db" \
  -e UPLOAD_DIR="/data/uploads" \
  mor-durian
```

### ถ้าจะใส่ LINE ID ลงปุ่ม "เพิ่มเพื่อน" บนเว็บ

ค่าพวก `NEXT_PUBLIC_*` ถูกฝังตอน build ไม่ใช่ตอนรัน ต้องส่งตอน build

```bash
docker build -t mor-durian \
  --build-arg NEXT_PUBLIC_LINE_ID="@xxxxxxx" \
  --build-arg NEXT_PUBLIC_LIFF_ID="1234567890-abcdefgh" .
```

---

## 3. Google Cloud Run

### ทำไมต้อง Docker ตรงนี้

Cloud Run รันได้เฉพาะ container เท่านั้น ส่วนวิธีที่ 1 กับ Render ไม่ต้องใช้ Docker ก็ได้

### ขั้นตอน

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>

# สิงคโปร์ ใกล้ไทยที่สุด
gcloud run deploy mor-durian \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars "LINE_CHANNEL_SECRET=xxx,LINE_CHANNEL_ACCESS_TOKEN=xxx,LIFF_ID=xxx,LIFF_CHANNEL_ID=xxx,WEB_BASE_URL=https://<url ที่ได้>"
```

### สามเรื่องที่พลาดบ่อย

**RAM ต้อง 2 GB ขึ้นไป** ค่าเริ่มต้นของ Cloud Run คือ 512 MB ซึ่งไม่พอสำหรับ
PyTorch แน่นอน จะตายตอนโหลดโมเดล

**เขียนไฟล์ได้เฉพาะ `/tmp`** ระบบไฟล์ของ Cloud Run เป็น read-only ยกเว้น `/tmp`
Dockerfile ตั้ง `DATABASE_URL` กับ `UPLOAD_DIR` ชี้ไป `/tmp` ไว้แล้ว
**ข้อมูลจะหายทุกครั้งที่ container รีสตาร์ต** ซึ่งรับได้สำหรับเดโม
ถ้าต้องการถาวรต้องย้ายไป Cloud SQL กับ Cloud Storage

**Cold start ประมาณ 30 วินาที** เพราะต้องโหลด PyTorch ถ้าปล่อยไว้ไม่มีคนใช้
Cloud Run จะลดเหลือศูนย์ instance คนแรกที่กดจะรอนาน
- สำหรับเดโม: เปิด `/health` อุ่นเครื่องก่อนส่งลิงก์ให้อาจารย์สัก 1 นาที
- ถ้าจะใช้จริง: `--min-instances=1` จะ warm ตลอด แต่มีค่าใช้จ่ายราว $10–18 ต่อเดือน

### หลัง deploy เสร็จ

เอา URL ที่ได้ไปตั้งที่

1. LINE → Webhook URL = `https://<url>/webhook` แล้วกด Verify
2. LIFF → Endpoint URL = `https://<url>/result`
3. Rich Menu ปุ่ม A → `https://<url>` (ดู `line-assets/HANDOFF.md`)
4. ถ้า `WEB_BASE_URL` ยังไม่ถูก ให้ `gcloud run services update` ใส่ค่าใหม่

---

## ตรวจว่าใช้งานได้จริง

ไล่ตามนี้หลัง deploy ทุกครั้ง

| # | ทำอะไร | ควรได้อะไร |
|---|---|---|
| 1 | เปิด `https://<url>/health` | `model_loaded: true` ถ้าใส่โมเดลแล้ว |
| 2 | เปิด `https://<url>/` | เห็นหน้าเว็บ มีโลโก้ |
| 3 | เปิด `https://<url>/diseases/` | เห็นรายการครบ 10 รายการ |
| 4 | LINE Developers กด **Verify** ที่ Webhook | Success |
| 5 | ทักแชทว่า `คลังความรู้` | ได้การ์ดเลื่อนข้าง |
| 6 | ส่งรูปใบทุเรียนเข้าไป | ขึ้นจุดไข่ปลา แล้วได้การ์ดผลตรวจ |
| 7 | กด "อ่านรายละเอียดเต็ม" ในการ์ด | เปิดหน้าผลตรวจในแอป LINE ได้ |

ถ้าข้อ 5 ตอบข้อความทั่วไปแทนการ์ด แปลว่า**ข้อความในปุ่ม Rich Menu ไม่ตรง**
กับคำที่ระบบรอรับ ดู `line-assets/HANDOFF.md`

---

## เพิ่มรายการใหม่ในคลังความรู้

ทุกอย่างอยู่ในไฟล์เดียว `app/data/diseases.json` ทั้งบอทและเว็บอ่านจากที่เดียวกัน

1. เพิ่ม object ใหม่ใน `"diseases"` โดย **`class_name` ต้องตรงกับชื่อคลาส
   ใน `config.yaml` → `data.classes` เป๊ะ ๆ** ไม่งั้นบอทจะทำนายคลาสที่ไม่มีคำอธิบาย
   (ระบบจะเตือนตอนเริ่มทำงานถ้าพบคลาสที่ยังไม่มีข้อมูล)
2. ใส่ `slug` เป็นตัวพิมพ์เล็กคั่นด้วยขีด จะกลายเป็น URL `/diseases/<slug>/`
3. ถ้ายังไม่มีเนื้อหา ตั้ง `"pending_expert_input": true` แล้วปล่อยช่องอื่นว่าง
   ระบบจะแสดงว่า "ข้อมูลกำลังจัดทำ" แทนที่จะแสดงข้อมูลเปล่า ๆ
4. **ห้ามใส่ตัวเลขอัตราการใช้สาร** ใน `chemical_options` ทุกช่อง `note`
   ต้องเป็น `"ใช้ตามอัตราบนฉลาก"` — มี test บังคับไว้ ถ้าใส่เลขจะ fail
5. รันตรวจ

```bash
python3 -m pytest tests/ -q
cd web && npm run build          # หน้าใหม่จะถูกสร้างอัตโนมัติ
```

---

## ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ |
|---|---|
| ทักอะไรไปบอทก็ตอบข้อความทั่วไป | ข้อความปุ่ม Rich Menu ไม่ตรงทุกตัวอักษร |
| ได้คำตอบสองครั้ง | ยังเปิด Auto-reply ใน OA Manager อยู่ |
| ส่งรูปแล้วได้ "ระบบยังไม่พร้อมใช้งาน" | ไม่มี `outputs/<model>/best.pt` หรือ `MODEL_NAME` ผิด |
| หน้าผลตรวจขึ้น "ดูผลตรวจนี้ไม่ได้" | เปิดนอกแอป LINE หรือ `LIFF_CHANNEL_ID` ผิดช่อง |
| Webhook กด Verify ไม่ผ่าน | URL ไม่ลงท้าย `/webhook` หรือเซิร์ฟเวอร์ยังไม่ขึ้น |
| `import torch` พัง | ลง torch ผิดแบบ ดูหัวข้อติดตั้งด้านบน |
| Cloud Run ตายตอนเริ่ม | RAM น้อยกว่า 2 GB |
