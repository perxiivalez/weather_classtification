import io
import numpy as np
from PIL import Image, ImageOps


IMG_SIZE = 32          
MAX_DECODE_SIDE = 512  

LUMA_W = np.array([0.299, 0.587, 0.114])


RGB2YCC = np.array([
    [0.299,      0.587,      0.114],     
    [-0.168736, -0.331264,  0.5],       
    [0.5,       -0.418688, -0.081312],   
])



def load_rgb(source):
    
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    im = Image.open(source)
    try:
        im.draft("RGB", (MAX_DECODE_SIDE, MAX_DECODE_SIDE))  
    except Exception:
        pass
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0



def brightness(img):
    
    P = img.reshape(-1, 3)        
    return float((P @ LUMA_W).mean())


def sky_mask(img):
    
    P = img.reshape(-1, 3)
    YCC = P @ RGB2YCC.T                        
    Y, Cb, Cr = YCC[:, 0], YCC[:, 1], YCC[:, 2]
    chroma = np.sqrt(Cb ** 2 + Cr ** 2)
    grey = (chroma < 0.06) & (Y > 0.08)         
    blue = (Cb > 0.03) & (Cb > Cr)               
    warm = (Cr > 0.06) & (Y > 0.20) & (Cr >= Cb) 
    return (grey | blue | warm).reshape(img.shape[:2])


def sky_ratio(img):
    return float(sky_mask(img).mean())


def roughness(img):
    g = img @ LUMA_W                            
    return float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)



VALID = dict(min_brightness=0.04, max_brightness=0.96, min_contrast=0.004,
             min_sky_ratio=0.12, max_roughness=0.15, min_side=16)


def validate(img):
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



def area_matrix(n_in, n_out):
    R = np.zeros((n_out, n_in))
    scale = n_in / n_out
    for i in range(n_out):
        lo, hi = i * scale, (i + 1) * scale
        for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            R[i, j] = min(j + 1, hi) - max(j, lo)
    return R / R.sum(axis=1, keepdims=True)


def resize(img, size=IMG_SIZE):
    
    H, W, C = img.shape
    Rh, Rw = area_matrix(H, size), area_matrix(W, size)
    tmp = np.tensordot(Rh, img, axes=(1, 0))          
    out = np.tensordot(tmp, Rw, axes=(1, 1))         
    return out.transpose(0, 2, 1)                    



def to_vector(img32):  
    return img32.reshape(-1)


def normalize(x, mean, std):
    return (x - mean) / std


def preprocess(img, mean=None, std=None, size=IMG_SIZE):
    v = to_vector(resize(img, size))
    return v if mean is None else normalize(v, mean, std)


FEATURE_MODES = ("raw32", "eng", "raw16+eng", "raw32+eng")


def engineered_features(X):
    
    N = len(X)
    Y = X @ LUMA_W                                        
    ycc = X @ RGB2YCC.T                                 
    Cb, Cr = ycc[..., 1], ycc[..., 2]
    sat = X.max(-1) - X.min(-1)
    chroma = np.sqrt(Cb ** 2 + Cr ** 2)
    sky = (((chroma < 0.06) & (Y > 0.08)) | ((Cb > 0.03) & (Cb > Cr)) | ((Cr > 0.06) & (Y > 0.20) & (Cr >= Cb)))
    R4 = area_matrix(Y.shape[1], 4)                       
    blocks = np.einsum("ik,nkl,jl->nij", R4, Y, R4).reshape(N, -1)     
    pct = np.percentile(Y.reshape(N, -1), [10, 50, 90], axis=1).T
    thirds = np.stack([Y[:, :11].mean((1, 2)), Y[:, 11:22].mean((1, 2)), Y[:, 22:].mean((1, 2))], 1)
    gx = np.abs(np.diff(Y, axis=2)).mean((1, 2))
    gy = np.abs(np.diff(Y, axis=1)).mean((1, 2))
    return np.concatenate([
        X.mean((1, 2)), X.std((1, 2)),                   
        Y.mean((1, 2))[:, None], Y.std((1, 2))[:, None],  
        pct,                                             
        sat.mean((1, 2))[:, None], Cb.mean((1, 2))[:, None], Cr.mean((1, 2))[:, None],  
        sky.mean((1, 2))[:, None], gx[:, None], gy[:, None],                            
        thirds, blocks], axis=1)                          


def build_features(X32, mode="raw32"):
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
