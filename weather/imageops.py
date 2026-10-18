"""
imageops.py - ขั้นตอนประมวลผลภาพด้วย NumPy (Linear Algebra)

PIL ใช้เพียง "ถอดรหัสไฟล์ JPEG/PNG" ให้เป็นอาร์เรย์ตัวเลขเท่านั้น
หลังจากนั้นทุกขั้นตอน (ความสว่าง, ตรวจท้องฟ้า, resize, normalize)
คำนวณด้วยการคูณเมทริกซ์/เวกเตอร์ของ NumPy ทั้งหมด
"""
import io
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------- ค่าคงที่
IMG_SIZE = 32          # ภาพถูกย่อเป็น 32 x 32 พิกเซล -> เวกเตอร์ยาว 32*32*3 = 3072
MAX_DECODE_SIDE = 512  # ลดขนาดตอนถอดรหัสไฟล์ใหญ่ ๆ (ประหยัดเวลา) ใช้เหมือนกันทั้ง train/serve

# เวกเตอร์น้ำหนักความสว่าง (luminance) :  Y = 0.299 R + 0.587 G + 0.114 B
LUMA_W = np.array([0.299, 0.587, 0.114])

# เมทริกซ์แปลงสี RGB -> (Y, Cb, Cr)  ขนาด 3x3  ใช้คูณกับเมทริกซ์พิกเซล (N x 3)
RGB2YCC = np.array([
    [0.299,      0.587,      0.114],      # Y  : ความสว่าง
    [-0.168736, -0.331264,  0.5],         # Cb : ความเป็นสีน้ำเงิน (-ส้ม/เหลือง)
    [0.5,       -0.418688, -0.081312],    # Cr : ความเป็นสีแดง/ส้ม (-เขียว)
])


# ---------------------------------------------------------------- 1) ภาพ -> เมทริกซ์
def load_rgb(source):
    """อ่านไฟล์/ไบต์ภาพ -> ค่า float ในช่วง [0,1] รูปร่าง (H, W, 3)
    คือ 'เมทริกซ์ 3 ชั้น' (Matrix Dimension H x W x 3) ที่แต่ละสมาชิกคือความเข้มของพิกเซล"""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    im = Image.open(source)
    try:
        im.draft("RGB", (MAX_DECODE_SIDE, MAX_DECODE_SIDE))  # JPEG: ถอดรหัสแบบย่อ (เร็วขึ้นมาก)
    except Exception:
        pass
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


# ---------------------------------------------------------------- 2) ตรวจสอบภาพเบื้องต้น
def brightness(img):
    """ค่าเฉลี่ยความสว่าง = mean( P @ w )  (P: N x 3, w: เวกเตอร์ 3 มิติ) -> สเกลาร์ 0..1"""
    P = img.reshape(-1, 3)            # แผ่เมทริกซ์ H x W x 3  เป็น N x 3  (N = H*W)
    return float((P @ LUMA_W).mean())


def sky_mask(img):
    """ตรวจหาพิกเซล 'คล้ายท้องฟ้า' ด้วยการคูณเมทริกซ์  YCC = P @ RGB2YCC^T
    ท้องฟ้ามี 3 แบบ:  เทา/ขาว (เมฆ,ฟ้าฝน) | น้ำเงิน (ฟ้าใส) | ส้ม-แดง (พระอาทิตย์ขึ้น/ตก)
    หญ้า ต้นไม้ พื้นดินสีเข้ม ฯลฯ จะไม่ถูกนับ"""
    P = img.reshape(-1, 3)
    YCC = P @ RGB2YCC.T                          # (N x 3) @ (3 x 3) = (N x 3)
    Y, Cb, Cr = YCC[:, 0], YCC[:, 1], YCC[:, 2]
    chroma = np.sqrt(Cb ** 2 + Cr ** 2)
    grey = (chroma < 0.06) & (Y > 0.08)          # เทา/ขาว
    blue = (Cb > 0.03) & (Cb > Cr)               # น้ำเงิน
    warm = (Cr > 0.06) & (Y > 0.20) & (Cr >= Cb)  # ส้ม/แดงสว่าง
    return (grey | blue | warm).reshape(img.shape[:2])


def sky_ratio(img):
    """สัดส่วนพิกเซลที่คล้ายท้องฟ้า  =  (1/N) * 1^T m   (m = เวกเตอร์ 0/1)"""
    return float(sky_mask(img).mean())


def roughness(img):
    """ความ 'หยาบ' ของภาพ = ค่าเฉลี่ยความต่างของพิกเซลที่อยู่ติดกัน (ดิฟเฟอเรนซ์ของเมทริกซ์ความสว่าง)
    วัดที่ความละเอียดเดิมของภาพ เพราะการย่อภาพจะเฉลี่ยสัญญาณรบกวนให้เรียบไปเอง
    ภาพถ่ายท้องฟ้าจริงเปลี่ยนค่อยเป็นค่อยไป (ค่าต่ำ)  ส่วนสัญญาณรบกวน/ลายถี่จะมีค่าสูง"""
    g = img @ LUMA_W                              # (H x W x 3) @ (3,) = (H x W)  ภาพระดับเทา
    return float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)


# เกณฑ์ตรวจสอบ (ปรับจากข้อมูลจริง ดู reports/validation_stats.json)
VALID = dict(min_brightness=0.04, max_brightness=0.96, min_contrast=0.004,
             min_sky_ratio=0.12, max_roughness=0.15, min_side=16)


def validate(img):
    """คืน (ผ่าน?, รายงาน dict) - ใช้ตัดสินว่ารูปนี้ประมวลผลต่อได้หรือไม่"""
    H, W, _ = img.shape
    b = brightness(img)
    contrast = float((img @ LUMA_W).std())
    s = sky_ratio(img)
    r = roughness(img)
    problems = []
    if min(H, W) < VALID["min_side"]:
        problems.append("รูปเล็กเกินไป")
    if b < VALID["min_brightness"]:
        problems.append("รูปมืดเกินไป")
    if b > VALID["max_brightness"]:
        problems.append("รูปสว่างจ้าเกินไป")
    if contrast < VALID["min_contrast"]:
        problems.append("รูปแทบไม่มีรายละเอียด (สีเดียวทั้งภาพ)")
    if s < VALID["min_sky_ratio"]:
        problems.append("ไม่พบท้องฟ้า/บรรยากาศในรูป")
    if r > VALID["max_roughness"]:
        problems.append("รูปมีสัญญาณรบกวน/ลายละเอียดถี่เกินไป ไม่น่าใช่ภาพถ่ายท้องฟ้า")
    report = dict(shape=[H, W, 3], brightness=round(b, 4), contrast=round(contrast, 4),
                  sky_ratio=round(s, 4), roughness=round(r, 4), problems=problems)
    return len(problems) == 0, report


# ---------------------------------------------------------------- 3) Resize ด้วยเมทริกซ์
def area_matrix(n_in, n_out):
    """สร้างเมทริกซ์ย่อขนาด R (n_out x n_in) แบบเฉลี่ยพื้นที่ (area averaging)
    แถวที่ i ของ R คือน้ำหนักของพิกเซลต้นทางที่ถูกเฉลี่ยเป็นพิกเซลปลายทางที่ i  (ผลรวมแถว = 1)"""
    R = np.zeros((n_out, n_in))
    scale = n_in / n_out
    for i in range(n_out):
        lo, hi = i * scale, (i + 1) * scale
        for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            R[i, j] = min(j + 1, hi) - max(j, lo)
    return R / R.sum(axis=1, keepdims=True)


def resize(img, size=IMG_SIZE):
    """ย่อภาพเป็น size x size ด้วยการคูณเมทริกซ์:   X' = R_h  X  R_w^T   (ทำทีละช่องสี)"""
    H, W, C = img.shape
    Rh, Rw = area_matrix(H, size), area_matrix(W, size)
    tmp = np.tensordot(Rh, img, axes=(1, 0))          # (size x H)(H x W x C) -> (size x W x C)
    out = np.tensordot(tmp, Rw, axes=(1, 1))          # (size x W x C)(W x size) -> (size x C x size)
    return out.transpose(0, 2, 1)                     # (size x size x C)


# ---------------------------------------------------------------- 4) Vectorize + Normalize
def to_vector(img32):
    """เมทริกซ์ 32x32x3 -> เวกเตอร์ยาว 3072 (flatten)"""
    return img32.reshape(-1)


def normalize(x, mean, std):
    """Standardization: x' = (x - mu) / sigma  (ทำทีละมิติ ใช้ mu, sigma จากชุด train เท่านั้น)"""
    return (x - mean) / std


def preprocess(img, mean=None, std=None, size=IMG_SIZE):
    """ภาพเมทริกซ์ -> เวกเตอร์ feature (ถ้าใส่ mean/std จะ normalize ให้ด้วย)"""
    v = to_vector(resize(img, size))
    return v if mean is None else normalize(v, mean, std)


# ---------------------------------------------------------------- 5) Feature vector (ทำงานเป็นชุด N ภาพ)
FEATURE_MODES = ("raw32", "eng", "raw16+eng", "raw32+eng")


def engineered_features(X):
    """เวกเตอร์คุณลักษณะ 36 มิติ จากภาพ 32x32 (X: N x 32 x 32 x 3) ด้วยการคูณเมทริกซ์/สถิติเวกเตอร์
    ความสว่าง, สีเฉลี่ย/ความเป็นสีน้ำเงิน/สีส้ม, สัดส่วนท้องฟ้า, ความหยาบ(เส้นฝน), แถบความสว่างบน-กลาง-ล่าง, และแผนที่ความสว่าง 4x4"""
    N = len(X)
    Y = X @ LUMA_W                                        # (N,32,32)  ภาพความสว่าง
    ycc = X @ RGB2YCC.T                                   # (N,32,32,3) แปลงสีด้วยเมทริกซ์ 3x3
    Cb, Cr = ycc[..., 1], ycc[..., 2]
    sat = X.max(-1) - X.min(-1)
    chroma = np.sqrt(Cb ** 2 + Cr ** 2)
    sky = (((chroma < 0.06) & (Y > 0.08)) | ((Cb > 0.03) & (Cb > Cr)) | ((Cr > 0.06) & (Y > 0.20) & (Cr >= Cb)))
    R4 = area_matrix(Y.shape[1], 4)                       # เมทริกซ์ย่อ 32 -> 4
    blocks = np.einsum("ik,nkl,jl->nij", R4, Y, R4).reshape(N, -1)      # R Y R^T : แผนที่ความสว่าง 4x4
    pct = np.percentile(Y.reshape(N, -1), [10, 50, 90], axis=1).T
    thirds = np.stack([Y[:, :11].mean((1, 2)), Y[:, 11:22].mean((1, 2)), Y[:, 22:].mean((1, 2))], 1)
    gx = np.abs(np.diff(Y, axis=2)).mean((1, 2))
    gy = np.abs(np.diff(Y, axis=1)).mean((1, 2))
    return np.concatenate([
        X.mean((1, 2)), X.std((1, 2)),                    # 3 + 3
        Y.mean((1, 2))[:, None], Y.std((1, 2))[:, None],  # 1 + 1
        pct,                                              # 3
        sat.mean((1, 2))[:, None], Cb.mean((1, 2))[:, None], Cr.mean((1, 2))[:, None],   # 3
        sky.mean((1, 2))[:, None], gx[:, None], gy[:, None],                             # 3
        thirds, blocks], axis=1)                          # 3 + 16 = 36


def build_features(X32, mode="raw32"):
    """ภาพ N x 32 x 32 x 3 -> เมทริกซ์คุณลักษณะ N x D ตามโหมดที่เลือก"""
    X32 = np.asarray(X32, dtype=np.float64)
    N = len(X32)
    parts = []
    if mode.startswith("raw32"):
        parts.append(X32.reshape(N, -1))
    if mode.startswith("raw16"):
        R = area_matrix(32, 16)
        parts.append(np.einsum("ik,nklc,jl->nijc", R, X32, R).reshape(N, -1))
    if "eng" in mode:
        parts.append(engineered_features(X32))
    return np.concatenate(parts, axis=1)
