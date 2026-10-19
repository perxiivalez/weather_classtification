import argparse
import json
import os
import numpy as np

from weather.data import DEFAULT_DATASET, ROOT, ensure_cache, read_config
from weather.imageops import build_features, FEATURE_MODES
from weather.cnn import HybridNet, gradient_check
from weather.data import load as load_cache
from weather.mlp import softmax


def augment(Ximg, rng, gain_range=0.25):
    n = len(Ximg)
    out = Ximg.copy()
    flip = rng.random(n) < 0.5
    out[flip] = out[flip][:, :, ::-1, :]
    gain = rng.uniform(1 - gain_range, 1 + gain_range, (n, 1, 1, 1))  
    contrast = rng.uniform(0.8, 1.2, (n, 1, 1, 1))              
    m = out.mean(axis=(1, 2, 3), keepdims=True)
    out = ((out - m) * contrast + m) * gain
    tint = rng.uniform(0.92, 1.08, (n, 1, 1, 3))                
    out = out * tint
    for i in range(n):                                          
        dx, dy = rng.integers(-3, 4, 2)
        out[i] = np.roll(out[i], (dy, dx), axis=(0, 1))
    out += rng.normal(0, 0.01, out.shape)
    return np.clip(out, 0, 1)


def metrics(net, XI, XF, y, k):
    P = net.predict_proba(XI, XF)
    pred = P.argmax(1)
    cm = np.zeros((k, k), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return dict(acc=float((pred == y).mean()), loss=float(net.loss(P, y)), cm=cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="โฟลเดอร์ dataset (1 โฟลเดอร์ย่อย = 1 คลาส)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache", "data32.npz"), help="ไฟล์ cache ภาพที่ย่อแล้ว")
    ap.add_argument("--rebuild", action="store_true", help="บังคับสร้าง cache ใหม่")
    ap.add_argument("--model", default="hybrid", choices=["hybrid", "cnn", "feat"],
                    help="hybrid = CNN + feature (ค่าเริ่มต้น) | cnn = เฉพาะ CNN | feat = เฉพาะ feature (เทียบ baseline)")
    ap.add_argument("--channels", type=int, nargs="+", default=[16, 32, 32], help="จำนวนฟิลเตอร์ของแต่ละบล็อก Conv")
    ap.add_argument("--fc-img", type=int, default=64, help="ขนาดชั้น FC หลัง CNN")
    ap.add_argument("--fc-feat", type=int, default=64, help="ขนาดชั้น FC ของสาขา feature")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gain", type=float, default=0.25,
                    help="ช่วงสุ่มความสว่างตอน augment (±) ; ลดลง (เช่น 0.1) ถ้า Cloudy/Storm Cloudy ปนกัน เพราะต่างกันที่ความมืด")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--feat", default="eng", choices=FEATURE_MODES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--final", action="store_true", help="ประเมินบน test และบันทึกโมเดล/รายงาน")
    ap.add_argument("--out", default=os.path.join(ROOT, "model"))
    a = ap.parse_args()

    if not os.path.isdir(a.dataset) and os.path.exists(a.cache) and not a.rebuild:
        print(f"ไม่พบโฟลเดอร์ dataset — ใช้ cache ที่มีอยู่ ({a.cache}) ซึ่งมีภาพ 32x32 และ split เดิมครบแล้ว", flush=True)
        d = load_cache(a.cache)
    else:
        d = ensure_cache(a.dataset, a.cache, rebuild=a.rebuild)
    Ximg, y, split = d["X"].astype(np.float64), d["y"], d["split"]
    CLASSES = [str(c) for c in d["classes"]]
    K = len(CLASSES)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"train={tr.sum()} val={va.sum()} test={te.sum()}  คลาส ({K}): {CLASSES}")
    counts = {c: [int(((y == i) & m).sum()) for m in (tr, va, te)] for i, c in enumerate(CLASSES)}
    print("จำนวนภาพต่อคลาส [train, val, test]:", counts)

    flat = lambda A: build_features(A, a.feat)                   
    mu, sd = flat(Ximg[tr]).mean(0), flat(Ximg[tr]).std(0) + 1e-6   
    imu, isd = Ximg[tr].mean((0, 1, 2)), Ximg[tr].std((0, 1, 2)) + 1e-6  
    norm = lambda A: ((A - imu) / isd, (flat(A) - mu) / sd)        
    (XIva, XFva), (XIte, XFte) = norm(Ximg[va]), norm(Ximg[te])

    net = HybridNet(flat(Ximg[:1]).shape[1], K, img_size=Ximg.shape[1], ch=a.channels, fc_img=a.fc_img, fc_feat=a.fc_feat,
                    use_cnn=a.model != "feat", use_feat=a.model != "cnn", l2=a.l2, dropout=a.dropout, seed=a.seed)
    print("โครงสร้าง:", net.describe(), flush=True)
    rng = np.random.default_rng(a.seed)
    hist, best = [], dict(acc=-1, loss=1e9, ep=0, state=None)
    idx_tr = np.where(tr)[0]
    for ep in range(1, a.epochs + 1):
        order = rng.permutation(idx_tr)
        aug = augment(Ximg[order], rng, a.gain)                          
        (XItr, XFtr), ytr = norm(aug), y[order]
        tl = 0.0
        for s in range(0, len(order), a.batch):
            sl = slice(s, s + a.batch)
            yb = ytr[sl]
            P, cache = net.forward(XItr[sl], XFtr[sl], train=True, rng=rng)
            tl += net.loss(P, yb) * len(yb)
            net.step(net.backward(yb, cache), a.lr)
        mv = metrics(net, XIva, XFva, y[va], K)
        hist.append(dict(ep=ep, train_loss=tl / len(order), val_loss=mv["loss"], val_acc=mv["acc"]))
        if mv["acc"] > best["acc"] or (mv["acc"] == best["acc"] and mv["loss"] < best["loss"]):
            best = dict(acc=mv["acc"], loss=mv["loss"], ep=ep, state={k: v.copy() for k, v in net.state().items()})
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep:3d}  train_loss {tl/len(order):.4f}  val_loss {mv['loss']:.4f}  val_acc {mv['acc']:.4f}", flush=True)
        if ep - best["ep"] >= a.patience:
            print(f"early stop ที่ epoch {ep} (ดีที่สุด epoch {best['ep']})")
            break

    net = HybridNet.from_state(best["state"])
    print(f"\nBEST  epoch {best['ep']}  val_acc {best['acc']:.4f}")
    if not a.final:
        return best["acc"]

    os.makedirs(a.out, exist_ok=True)
    mt = metrics(net, XIte, XFte, y[te], K)
    mt_tr = metrics(net, *norm(Ximg[tr]), y[tr], K)
    Tgrid = np.round(np.arange(0.5, 4.01, 0.05), 2)
    LGva = net.logits(XIva, XFva).astype(np.float64)
    nll = [float(-np.log(softmax(LGva / T)[np.arange(len(LGva)), y[va]] + 1e-12).mean()) for T in Tgrid]
    T = float(Tgrid[int(np.argmin(nll))])

    def ece(P, yy, bins=10):          
        conf, pred = P.max(1), P.argmax(1)
        e = 0.0
        for lo in np.linspace(0, 1, bins, endpoint=False):
            m = (conf > lo) & (conf <= lo + 1 / bins)
            if m.any():
                e += m.mean() * abs((pred[m] == yy[m]).mean() - conf[m].mean())
        return float(e)
    calib = dict(temperature=T, val_nll_before=float(nll[int(np.where(Tgrid == 1.0)[0][0])]), val_nll_after=float(min(nll)),
                 test_ece_before=ece(net.predict_proba(XIte, XFte), y[te]), test_ece_after=ece(net.predict_proba(XIte, XFte, T), y[te]))
    cm = mt["cm"]
    rec = cm.diagonal() / cm.sum(1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    rep = dict(test_acc=mt["acc"], val_acc=best["acc"], train_acc=mt_tr["acc"], best_epoch=best["ep"],
               n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
               per_class={c: dict(precision=float(prec[i]), recall=float(rec[i]), support=int(cm.sum(1)[i]))
                          for i, c in enumerate(CLASSES)},
               class_counts=counts, calibration=calib, confusion_matrix=cm.tolist(), classes=CLASSES,
               hyper={k: v for k, v in vars(a).items() if k not in ("dataset", "cache", "out")}, model=net.describe(), gradient_check_max_rel_error=float(gradient_check()))
    rain = [c for c in read_config(a.dataset)["rain_classes"] if c in CLASSES]   
    np.savez(os.path.join(a.out, "weights.npz"), mean=mu, std=sd, img_mean=imu, img_std=isd, classes=np.array(CLASSES), feat=np.array(a.feat),
             temp=np.array(T), rain_classes=np.array(rain, dtype="<U32"), **best["state"])
    json.dump(rep, open(os.path.join(a.out, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
    print(json.dumps({k: rep[k] for k in ("test_acc", "val_acc", "train_acc", "per_class", "class_counts",
                                          "calibration", "gradient_check_max_rel_error")}, indent=2, ensure_ascii=False))
    print("confusion matrix (แถว=จริง, คอลัมน์=ทำนาย):\n", cm)
    return mt["acc"]


if __name__ == "__main__":
    main()
