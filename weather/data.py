"""data.py - อ่านชุดข้อมูลจากโฟลเดอร์ dataset/ แล้วแปลงภาพเป็นเวกเตอร์ + แบ่ง train/val/test

โครงสร้างโฟลเดอร์ (เพิ่ม/ลบ/ย้ายรูปได้อิสระ แล้วเทรนใหม่ได้เลย):

    dataset/
      classes.json        <- (ไม่บังคับ) ลำดับคลาส, ชื่อเรียกอื่น (aliases), คลาสที่นับเป็น "ฝน"
      Cloudy/   *.jpg     <- 1 โฟลเดอร์ = 1 คลาส  (ชื่อโฟลเดอร์ = ชื่อคลาส, มีโฟลเดอร์ย่อยข้างในได้)
      Rain/     *.jpg
      Storm Cloudy/ *.jpg
      ...
      cloudy1.jpg         <- (รองรับ) รูปที่วางหลวม ๆ ในโฟลเดอร์ dataset เลย จะจับคลาสจากคำนำหน้าชื่อไฟล์

เพิ่มคลาสใหม่ = สร้างโฟลเดอร์ใหม่ใส่รูป (จำนวนคลาสของโมเดลปรับเองอัตโนมัติ)
"""
import glob
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import imageops as ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(ROOT, "dataset")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_PER_CLASS = 5      # คลาสที่มีรูปน้อยกว่านี้จะถูกข้าม (แบ่ง train/val/test ไม่ได้) พร้อมเตือน


def _norm(s):
    """ทำชื่อให้เทียบกันได้: ตัวพิมพ์เล็ก ไม่สนช่องว่าง/ขีด  ('Storm_Cloudy' == 'storm cloudy')"""
    return re.sub(r"[\s_\-]+", "", str(s)).lower()


def read_config(dataset_dir):
    cfg = dict(classes=None, aliases={}, rain_classes=["Rain", "Storm Cloudy"])
    p = os.path.join(dataset_dir, "classes.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    return cfg


def _is_image(f):
    return os.path.isfile(f) and os.path.splitext(f)[1].lower() in IMG_EXT


def discover(dataset_dir):
    """สแกน dataset -> (classes, items) ; items = [(path, class_index)]"""
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"ไม่พบโฟลเดอร์ dataset: {dataset_dir}")
    cfg = read_config(dataset_dir)
    alias = {_norm(k): v for k, v in cfg["aliases"].items()}
    by_class = {}                                   # ชื่อคลาส -> [ไฟล์]

    # (1) โฟลเดอร์ = คลาส
    for entry in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, entry)
        if not os.path.isdir(d) or entry.startswith((".", "__")):
            continue
        name = alias.get(_norm(entry), entry)
        files = [f for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)) if _is_image(f)]
        by_class.setdefault(name, []).extend(files)

    # (2) รูปที่วางหลวมในโฟลเดอร์ dataset: จับคลาสจากคำนำหน้าชื่อไฟล์ (เช่น rain12.jpg, storm_cloudy3.png)
    loose = [f for f in sorted(glob.glob(os.path.join(dataset_dir, "*"))) if _is_image(f)]
    if loose:
        known = {_norm(c): c for c in (cfg["classes"] or [])}
        known.update({_norm(c): c for c in by_class})
        known.update({k: v for k, v in alias.items()})
        prefixes = sorted(known, key=len, reverse=True)       # คำนำหน้ายาวสุดก่อน (rainy ก่อน rain)
        unmatched = 0
        for f in loose:
            n = _norm(os.path.splitext(os.path.basename(f))[0])
            hit = next((p for p in prefixes if n.startswith(p)), None)
            if hit is None:
                unmatched += 1
            else:
                by_class.setdefault(known[hit], []).append(f)
        if unmatched:
            print(f"[คำเตือน] มีรูปในโฟลเดอร์ dataset {unmatched} ไฟล์ที่ชื่อไม่ตรงคลาสใด ๆ จึงข้ามไป", flush=True)

    # ลำดับคลาส: ตาม classes.json ก่อน (คลาสที่เหลือต่อท้ายตามตัวอักษร); ข้ามคลาสที่ไม่มีรูป
    wanted = list(cfg["classes"] or [])
    for c in wanted:
        if not by_class.get(c):
            print(f"[คำเตือน] คลาส '{c}' อยู่ใน classes.json แต่ยังไม่มีรูป จึงยังไม่นำมาเทรน", flush=True)
    for c in sorted(by_class):
        if 0 < len(by_class[c]) < MIN_PER_CLASS:
            print(f"[คำเตือน] โฟลเดอร์/คลาส '{c}' มีรูปแค่ {len(by_class[c])} (< {MIN_PER_CLASS}) จึงข้าม — "
                  f"ถ้าเป็นชื่อสะกดต่างของคลาสอื่น ให้เพิ่มใน aliases ของ classes.json", flush=True)
            by_class[c] = []
    classes = [c for c in wanted if by_class.get(c)] + sorted(c for c in by_class if c not in wanted and by_class[c])
    if len(classes) < 2:
        raise SystemExit(f"ต้องมีอย่างน้อย 2 คลาสที่มีรูปใน {dataset_dir} (พบ {len(classes)})")
    items = [(f, i) for i, c in enumerate(classes) for f in by_class[c]]
    return classes, items, cfg


def fingerprint(classes, items, cfg):
    """ลายนิ้วมือของ dataset: เปลี่ยนเมื่อมีการเพิ่ม/ลบ/แก้ไขรูป หรือแก้ classes.json -> ใช้ตัดสินใจสร้าง cache ใหม่"""
    h = hashlib.sha1(json.dumps([classes, cfg], sort_keys=True, ensure_ascii=False).encode())
    for p, y in items:
        st = os.stat(p)
        h.update(f"{os.path.relpath(p)}|{y}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _process(path):
    try:
        img = ops.load_rgb(path)
        ok, rep = ops.validate(img)
        return ops.resize(img, ops.IMG_SIZE).astype(np.float32), ok, rep
    except Exception:
        return None                                  # ไฟล์รูปเสีย


def _split(y, k, seed):
    """แบ่ง 70/15/15 แบบ stratified ตามคลาส"""
    rng = np.random.default_rng(seed)
    split = np.empty(len(y), dtype="<U5")
    for c in range(k):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_va = max(1, int(round(.15 * n))) if n >= 3 else 0
        n_te = max(1, int(round(.15 * n))) if n >= 3 else 0
        n_tr = n - n_va - n_te
        split[idx[:n_tr]], split[idx[n_tr:n_tr + n_va]], split[idx[n_tr + n_va:]] = "train", "val", "test"
    return split


def build(dataset_dir, out_npz, seed=42, workers=None):
    classes, items, cfg = discover(dataset_dir)
    fp = fingerprint(classes, items, cfg)
    print(f"พบภาพทั้งหมด {len(items)} ภาพ ใน {len(classes)} คลาส: " +
          ", ".join(f"{c}={sum(1 for _, y in items if y == i)}" for i, c in enumerate(classes)), flush=True)
    workers = workers or min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_process, [p for p, _ in items], chunksize=16))
    bad = [p for (p, _), r in zip(items, res) if r is None]
    if bad:
        print(f"[คำเตือน] อ่านรูปไม่ได้ {len(bad)} ไฟล์ (ข้าม): " + ", ".join(os.path.basename(b) for b in bad[:5]) +
              (" ..." if len(bad) > 5 else ""), flush=True)
    good = [i for i, r in enumerate(res) if r is not None]
    items = [items[i] for i in good]
    res = [res[i] for i in good]
    X = np.stack([r[0] for r in res])                      # (N, 32, 32, 3)
    ok = np.array([r[1] for r in res])
    reps = [r[2] for r in res]
    y = np.array([i[1] for i in items])
    paths = np.array([os.path.relpath(i[0], dataset_dir) for i in items])

    # ---- ลบภาพซ้ำ (ซ้ำเมื่อเวกเตอร์ 32x32 เหมือนกันหลังปัดค่า)
    # ถ้าภาพเดียวกันอยู่ต่างคลาสกัน (ติดป้ายขัดแย้ง) จะตัดทิ้งทั้งหมดเพราะสอนโมเดลสับสน
    groups = {}
    for i, x in enumerate(X):
        groups.setdefault(hashlib.md5(np.round(x * 31).astype(np.uint8).tobytes()).hexdigest(), []).append(i)
    keep = np.zeros(len(X), bool)
    conflicts = 0
    for idxs in groups.values():
        if len({int(y[i]) for i in idxs}) > 1:
            conflicts += len(idxs)
            continue
        keep[idxs[0]] = True
    print(f"ลบภาพซ้ำ {len(X) - int(keep.sum()) - conflicts} ภาพ"
          + (f" และตัดภาพที่ซ้ำกันข้ามคลาส (ป้ายขัดแย้ง) {conflicts} ภาพ" if conflicts else "")
          + f" เหลือ {int(keep.sum())}", flush=True)
    X, y, paths, ok = X[keep], y[keep], paths[keep], ok[keep]
    reps = [r for r, k in zip(reps, keep) if k]

    split = _split(y, len(classes), seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez_compressed(out_npz, X=X, y=y, paths=paths, split=split, ok=ok,
                        classes=np.array(classes), fingerprint=np.array(fp))
    with open(os.path.join(os.path.dirname(os.path.abspath(out_npz)), "validation_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dict(classes=classes, per_image=[dict(path=str(p), cls=classes[int(c)], **r)
                                                   for p, c, r in zip(paths, y, reps)]), f, ensure_ascii=False)
    return load(out_npz)


def ensure_cache(dataset_dir, cache_npz, rebuild=False, seed=42):
    """ใช้ cache เดิมถ้า dataset ไม่เปลี่ยน ; ถ้ามีการเพิ่ม/ลบ/แก้รูป (หรือสั่ง rebuild) จะสร้างใหม่ให้อัตโนมัติ"""
    classes, items, cfg = discover(dataset_dir)
    fp = fingerprint(classes, items, cfg)
    if not rebuild and os.path.exists(cache_npz):
        d = load(cache_npz)
        if str(d.get("fingerprint", "")) == fp:
            print(f"ใช้ cache เดิม ({cache_npz}) — dataset ไม่เปลี่ยน", flush=True)
            return d
        print("dataset เปลี่ยนไปจากตอนสร้าง cache -> สร้าง cache ใหม่", flush=True)
    return build(dataset_dir, cache_npz, seed=seed)


def load(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    # python -m weather.data [โฟลเดอร์ dataset] [ไฟล์ปลายทาง .npz]
    ds = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "cache", "data32.npz")
    build(ds, out)
