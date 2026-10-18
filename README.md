# พยากรณ์โอกาสเกิดฝนจากภาพท้องฟ้า — CNN + Feature Hybrid (NumPy)

จำแนกภาพถ่ายเป็นสภาวะท้องฟ้า (Cloudy, Rain, Shine, Sunrise, Storm Cloudy — จำนวนคลาสตามโฟลเดอร์ใน `dataset/`) และบอกโอกาสเกิดฝนเป็นเปอร์เซ็นต์
โมเดลเป็น **ไฮบริด**: CNN อ่านภาพ 32×32 โดยตรง **ควบคู่กับ** เวกเตอร์คุณลักษณะ 36 มิติเดิม แล้วรวมกันตัดสินที่ชั้นสุดท้าย
เขียนด้วย NumPy (conv/pool/backprop/Adam เขียนเอง) ไม่ต้องติดตั้ง PyTorch/TensorFlow (PIL ใช้เพียงถอดรหัสไฟล์ JPG/PNG)

## วิธีใช้

```bash
pip install -r requirements.txt

# เรียก API  (พอร์ต 8000, เปิดหน้าเว็บอัปโหลดที่ http://localhost:8000/)
python api.py
curl -F "image=@sky.jpg" http://localhost:8000/predict
curl -F "image=@sky.jpg" "http://localhost:8000/predict?trace=1"   # แสดงมิติเมทริกซ์ทุกขั้นตอน

# ดูการคำนวณทุกขั้นตอนของภาพหนึ่งภาพในเทอร์มินัล
python demo_trace.py sky.jpg
```

ตัวอย่างผลลัพธ์ `/predict`

```json
{"status":"ok","predicted_class":"Rain","confidence_percent":84.5,"rain_probability_percent":84.5,
 "probabilities":{"Cloudy":15.3,"Rain":84.5,"Shine":0.2,"Sunrise":0.0},
 "validation":{"brightness":0.3847,"sky_ratio":0.9919,"contrast":0.1565,"roughness":0.0097,"problems":[]}}
```
ภาพที่ไม่ผ่านการตรวจสอบ (มืด/สว่างเกิน/สีเดียว/ไม่มีท้องฟ้า/สัญญาณรบกวน) จะตอบ HTTP 422 พร้อมเหตุผล
ไฟล์ที่ไม่ใช่รูปหรือไม่แนบไฟล์ตอบ 400

## Dataset และการเทรนใหม่

รูปทั้งหมดอยู่ในโฟลเดอร์ `dataset/` — **1 โฟลเดอร์ย่อย = 1 คลาส** (ดูรายละเอียดใน `dataset/README.md`)

```
dataset/
  classes.json     (ไม่บังคับ: ลำดับคลาส / aliases ชื่อสะกดอื่น / rain_classes)
  Cloudy/  Rain/  Shine/  Sunrise/  Storm Cloudy/
```

```bash
# แก้ dataset (เพิ่ม/ลบ/ย้ายรูป หรือสร้างโฟลเดอร์คลาสใหม่) แล้วรัน:
python train.py --final
# (ระบบตรวจว่ารูปเปลี่ยนหรือไม่ แล้วสร้าง cache ใหม่เอง ; บังคับสร้างใหม่ด้วย --rebuild)
# ใช้ dataset ที่อื่น:  python train.py --dataset /path/to/dataset --final
# ถ้า Cloudy กับ Storm Cloudy ยังปนกัน ลองลดการสุ่มความสว่างตอน augment:  --gain 0.1
```

ค่าเริ่มต้นของ `train.py`: `--model hybrid --channels 16 32 32 --fc-img 64 --fc-feat 64 --feat eng --dropout 0.3 --l2 1e-3 --seed 1`
เทียบสถาปัตยกรรมได้ด้วย `--model feat` (เฉพาะ feature ≈ MLP เดิม) / `--model cnn` (เฉพาะ CNN) / `--model hybrid`
ถ้าไม่พบโฟลเดอร์ `dataset/` แต่มี `cache/data32.npz` จะเทรนจาก cache ได้เลย
เทรนเสร็จจะได้ `model/weights.npz` (เก็บรายชื่อคลาสไว้ในไฟล์ `api.py` อ่านเองอัตโนมัติ) และ `model/metrics.json`
(confusion matrix, precision/recall รายคลาส, จำนวนภาพต่อคลาส)

**โอกาสเกิดฝน** = `min(1, Σ w_k · p_k)` คือรวมความน่าจะเป็นของ **Rain** และ **Storm Cloudy** แบบถ่วงน้ำหนัก
(ค่าเริ่มต้น Rain = 1.0, Storm Cloudy = 0.7) ผลลัพธ์ API จะมี `rain_breakdown` บอกว่าแต่ละคลาสช่วยเพิ่มเท่าไร
ปรับน้ำหนักได้ **โดยไม่ต้องเทรนใหม่** ที่ `model/rain_config.json`:

```json
{"rain_weights": {"Rain": 1.0, "Storm Cloudy": 0.8}}
```

(ใส่ 1.0 ทั้งคู่ = บวกเต็ม ๆ) ชื่อคลาสไม่สนตัวพิมพ์/ช่องว่าง/ขีด และจะใช้เฉพาะคลาสที่โมเดลมีจริง

## สถาปัตยกรรมโมเดล (ไฟล์ `weather/cnn.py`)

```
ภาพ 32×32×3 ─► [Conv3×3 → ReLU → MaxPool2] ×3 (16→32→32 ช่อง) ─► flatten 512 ─► FC 64 + ReLU ─┐
                                                                                                ├─► concat 128 ─► Dropout ─► FC 5 ─► softmax(÷T)
feature 36 มิติ (normalize) ─────────────────────────────────────► FC 64 + ReLU ───────────────┘
```
พารามิเตอร์ทั้งหมด ≈ 50,000 ตัว · Conv ทำด้วย im2col เป็นการคูณเมทริกซ์ `cols (N·H·W × C·9) @ W (C·9 × F)`
Backward: `dW = colsᵀ·dZ`, `dcols = dZ·Wᵀ` แล้ว col2im ; MaxPool ส่งเกรเดียนต์ไปยังตำแหน่งค่ามากสุด

## ลำดับการทำงาน และทฤษฎี Vector & Matrix ที่ใช้ (ไฟล์ `weather/imageops.py`, `cnn.py`, `pipeline.py`)

| ขั้นตอน | การคำนวณ |
|---|---|
| ภาพ → เมทริกซ์ | อาร์เรย์ H×W×3 ค่า 0..1 (Matrix Dimension) |
| ความสว่างเฉลี่ย | แผ่เป็น P (N×3) แล้ว `mean(P @ w)`, w=[0.299,0.587,0.114] |
| ตรวจท้องฟ้า | `YCC = P @ Tᵀ` (T เมทริกซ์แปลงสี 3×3) แล้วนับพิกเซลเทา/น้ำเงิน/ส้ม |
| ตรวจสอบภาพ | ความสว่าง, คอนทราสต์, สัดส่วนท้องฟ้า, ความหยาบ ต้องอยู่ในเกณฑ์ |
| Resize | `X' = R_h · X · R_wᵀ` โดย R คือเมทริกซ์เฉลี่ยพื้นที่ (ผลรวมแต่ละแถว = 1) |
| สาขา feature | เวกเตอร์ 36 มิติ (สีเฉลี่ย/ส่วนเบี่ยงเบน, ความสว่าง, เปอร์เซนไทล์, ความอิ่มสี, Cb/Cr, สัดส่วนท้องฟ้า, ความหยาบ, ความสว่างบน-กลาง-ล่าง, แผนที่ความสว่าง 4×4) → `x' = (x−μ)/σ` → `FC+ReLU` |
| สาขา CNN | ภาพ normalize ต่อช่องสี → Conv3×3+ReLU+MaxPool ×3 → flatten → `FC+ReLU` |
| รวมสาขา | `h = [A_cnn ‖ A_feat]` (1×128) → `Z = hW + b` (128→5) |
| Softmax | `p_k = exp(z_k/T) / Σ exp(z_j/T)` (T ปรับจาก validation) |
| Backprop | chain rule แบบเมทริกซ์ ทุกชั้น รวม conv/pool/concat (ดู `cnn.py`) |
| Optimizer | Adam, L2, dropout, early stopping (เขียนเอง) |

ตรวจ Backprop ด้วย gradient check: ความคลาดเคลื่อนสัมพัทธ์ ≈ 5×10⁻¹⁰ (`python -m weather.cnn`)

## ผลลัพธ์ (โมเดลไฮบริด 5 คลาส)

ข้อมูล: 1,520 ภาพหลังลบภาพซ้ำ แบ่ง train 1,064 / val 228 / test 228 (stratified, split เดียวกับโมเดลเดิม)

| ตัวชี้วัด | MLP เดิม (feature) | **CNN + feature** |
|---|---|---|
| **Test accuracy** | 90.4% | **93.0%** (212/228) |
| Validation accuracy | 91.2% | 93.9% |
| Rain precision / recall | 88.7 / 92.2 | 92.0 / 90.2 |
| Storm Cloudy precision / recall | 83.6 / 92.7 | 92.7 / 92.7 |
| Shine precision / recall | 93.2 / 87.3 | 90.8 / 93.7 |
| Sunrise precision / recall | 96.4 / 100 | 100 / 98.1 |
| Cloudy precision / recall (test มีแค่ 6 ภาพ) | 0 / 0 | 66.7 / 66.7 |

เปรียบเทียบสถาปัตยกรรมบน **validation** (2 seed, ยังไม่แตะ test): feature อย่างเดียว 88.6 / 90.4 · CNN อย่างเดียว 93.0 / 94.3 · ไฮบริด 93.9 / 92.5
→ การเพิ่ม CNN ดีขึ้น ≈ 3–4 จุดชัดเจน แต่ **ไฮบริดยังไม่ดีกว่า CNN อย่างเดียวอย่างมีนัยสำคัญ** (ความต่างอยู่ในระดับความแกว่งของ validation 228 ภาพ)

ตัวเลขทั้งหมด: `model/metrics.json` · `reports/training_report.png` เป็นกราฟของโมเดลรุ่นเก่า ยังไม่ได้สร้างใหม่

## ข้อจำกัดที่ควรรู้

1. **คลาส Cloudy มีแค่ 39 ภาพทั้งหมด** (train 27 / val 6 / test 6) ผล precision/recall ของ Cloudy จึงไม่น่าเชื่อถือทางสถิติ
   (6 ภาพ: ถูก 4) ควรเพิ่มรูป Cloudy ก่อนเป็นอย่างแรก เพราะจะช่วยได้มากกว่าการปรับโมเดล
2. **ตัวเลข % คือความน่าจะเป็นที่ภาพนี้ "ดูเป็นฝน"** ไม่ใช่การพยากรณ์อุตุนิยมวิทยา ภาพชุด Cuaca ส่วนใหญ่เป็นแค่ฟ้ามืด ไม่เห็นเม็ดฝน
3. **ข้อมูลน้อย (≈1,060 ภาพเทรน)** และมาจากแหล่งจำกัด ผลกับรูปถ่ายจากโทรศัพท์จริงอาจต่ำกว่านี้ ควรทดสอบกับรูปที่ถ่ายเองก่อนใช้งานจริง
4. **ความโปร่งใสของการเลือกโมเดล**: สถาปัตยกรรม CNN+feature และ hyper-parameter ที่ใช้ตัดสินใจจาก validation เท่านั้น และรัน test ครั้งเดียวด้วย seed 1 (ค่าเริ่มต้นเดิม)
   แต่ split test ชุดนี้เป็นชุดเดียวกับที่รายงานโมเดลเดิมไว้ก่อนหน้า จึงไม่ใช่ test ที่ไม่เคยถูกเห็นเลย · test 228 ภาพ ช่วงความเชื่อมั่น 95% ≈ ±3 จุด
5. ผลเทียบสถาปัตยกรรมใช้เพียง 2 seed ความต่างระหว่างไฮบริดกับ CNN อย่างเดียวจึงสรุปไม่ได้
6. ตัวตรวจท้องฟ้าเป็นกฎสีแบบง่าย ใช้เป็นด่านกรองภาพเสียชัดเจน ไม่ได้รับประกันว่าทุกภาพที่ผ่านเป็นภาพท้องฟ้า
# weather_classtification
