"""
cnn.py - โมเดลไฮบริด: CNN (อ่านภาพ 32x32x3) ควบคู่กับเวกเตอร์คุณลักษณะ 36 มิติ  (NumPy)

            ภาพ 32x32x3 ──► [Conv3x3 → ReLU → MaxPool2] x B ──► flatten ──► FC_img+ReLU ─┐
                                                                                         ├─► concat ─► Dropout ─► FC_out ─► softmax
   คุณลักษณะ 36 มิติ (x') ───────────────────────────────────► FC_feat+ReLU ─────────────┘

  สาขา CNN เรียนรู้ลวดลาย/โครงสร้างเมฆ-เส้นฝนจากพิกเซลโดยตรง ส่วนสาขา feature คงความรู้เดิม
  (ความสว่าง สัดส่วนท้องฟ้า ความหยาบ ฯลฯ) แล้วให้ชั้นสุดท้ายตัดสินจากทั้งสองอย่างพร้อมกัน

  Convolution ทำด้วย im2col เพื่อให้เป็นการคูณเมทริกซ์ (สอดคล้องกับ mlp.py):
      cols (N*H*W  x  C*9)  @  W (C*9 x F)  =  Z (N*H*W x F)
  Backward ของ conv:  dW = colsᵀ dZ ,  dcols = dZ Wᵀ  แล้ว "col2im" บวกกลับเข้าตำแหน่งพิกเซลเดิม
  Backward ของ MaxPool: ส่งเกรเดียนต์ไปยังตำแหน่งที่เป็นค่ามากสุดของแต่ละหน้าต่าง 2x2
"""
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .mlp import softmax, relu


# ------------------------------------------------------------------ Conv 3x3 (stride 1, padding 1)
def conv3x3_forward(x, W, b):
    """x: (N,H,W,C)  W: (C*9, F)  b: (1,F)  ->  Z: (N,H,W,F), cols (เก็บไว้ใช้ตอน backprop)"""
    N, H, Wd, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols = sliding_window_view(xp, (3, 3), axis=(1, 2)).reshape(N * H * Wd, C * 9)   # im2col
    Z = (cols @ W + b).reshape(N, H, Wd, -1)
    return Z, cols


def conv3x3_backward(dZ, cols, W, x_shape):
    N, H, Wd, C = x_shape
    d2 = dZ.reshape(-1, dZ.shape[-1])                      # (N*H*W x F)
    dW = cols.T @ d2                                       # (C*9 x N*H*W)(N*H*W x F)
    db = d2.sum(axis=0, keepdims=True)
    dcols = (d2 @ W.T).reshape(N, H, Wd, C, 3, 3)          # (N*H*W x F)(F x C*9)
    dxp = np.zeros((N, H + 2, Wd + 2, C), dtype=dcols.dtype)
    for i in range(3):                                     # col2im
        for j in range(3):
            dxp[:, i:i + H, j:j + Wd, :] += dcols[:, :, :, :, i, j]
    return dxp[:, 1:-1, 1:-1, :], dW, db


# ------------------------------------------------------------------ MaxPool 2x2
def maxpool_forward(A):
    N, H, W, C = A.shape
    r = A.reshape(N, H // 2, 2, W // 2, 2, C)
    out = r.max(axis=(2, 4))
    mask = (r == out[:, :, None, :, None, :])
    mask = mask / mask.sum(axis=(2, 4), keepdims=True)     # ถ้าค่าเท่ากันหลายตำแหน่ง แบ่งเกรเดียนต์เท่า ๆ กัน
    return out, mask.astype(A.dtype)


def maxpool_backward(dout, mask):
    N, h, w, C = dout.shape
    return (mask * dout[:, :, None, :, None, :]).reshape(N, h * 2, w * 2, C)


# ------------------------------------------------------------------ เครือข่ายไฮบริด
class HybridNet:
    def __init__(self, n_feat, n_classes, img_size=32, ch=(16, 32, 32), fc_img=64, fc_feat=64,
                 use_cnn=True, use_feat=True, l2=1e-3, dropout=0.0, seed=0, dtype=np.float32):
        assert use_cnn or use_feat
        self.cfg = dict(n_feat=int(n_feat), n_classes=int(n_classes), img_size=int(img_size), ch=[int(c) for c in ch],
                        fc_img=int(fc_img), fc_feat=int(fc_feat), use_cnn=bool(use_cnn), use_feat=bool(use_feat))
        self.l2, self.dropout, self.dtype = l2, dropout, dtype
        self.ch, self.use_cnn, self.use_feat = list(ch), use_cnn, use_feat
        rng = np.random.default_rng(seed)
        he = lambda m, n: rng.normal(0, np.sqrt(2.0 / m), (m, n)).astype(dtype)     # He initialization
        zeros = lambda n: np.zeros((1, n), dtype)
        P, width = {}, 0
        if use_cnn:
            c_in, s = 3, img_size
            for l, c_out in enumerate(self.ch):
                P[f"cW{l}"], P[f"cb{l}"] = he(c_in * 9, c_out), zeros(c_out)
                c_in, s = c_out, s // 2
            self.flat_dim = s * s * c_in
            P["iW"], P["ib"] = he(self.flat_dim, fc_img), zeros(fc_img)
            width += fc_img
        if use_feat:
            P["fW"], P["fb"] = he(n_feat, fc_feat), zeros(fc_feat)
            width += fc_feat
        P["oW"], P["ob"] = he(width, n_classes), zeros(n_classes)
        self.P, self.width, self._adam = P, width, None

    # ------------------------------------------------------------ Forward
    def forward(self, Ximg, F, train=False, rng=None):
        """Ximg: (N,S,S,3) ภาพที่ normalize แล้ว ; F: (N,D) คุณลักษณะที่ normalize แล้ว -> (P, cache)
        cache['trace'] เก็บมิติของทุกชั้นไว้แสดงผล (ไม่มีผลต่อการคำนวณ)"""
        P, c, parts, tr = self.P, {}, [], []
        if self.use_cnn:
            a, c["conv"] = Ximg.astype(self.dtype, copy=False), []
            for l in range(len(self.ch)):
                Z, cols = conv3x3_forward(a, P[f"cW{l}"], P[f"cb{l}"])
                A = relu(Z)
                pooled, mask = maxpool_forward(A)
                c["conv"].append(dict(cols=cols, Z=Z, in_shape=a.shape, mask=mask))
                tr.append((f"cnn_block_{l+1}", f"Conv3x3 {a.shape[1:]} -> {Z.shape[1:]} -> ReLU -> MaxPool2 -> {pooled.shape[1:]}",
                           int((A > 0).sum()), int(A.size)))
                a = pooled
            flat = a.reshape(len(a), -1)
            Zi = flat @ P["iW"] + P["ib"]
            Ai = relu(Zi)
            c["img"] = dict(flat=flat, Z=Zi, a_shape=a.shape)
            tr.append(("cnn_fc", f"flatten {list(a.shape[1:])} -> {list(flat.shape)} @ {list(P['iW'].shape)} -> ReLU -> {list(Ai.shape)}", None, None))
            parts.append(Ai)
        if self.use_feat:
            Zf = F.astype(self.dtype, copy=False) @ P["fW"] + P["fb"]
            Af = relu(Zf)
            c["feat"] = dict(F=F.astype(self.dtype, copy=False), Z=Zf)
            tr.append(("feature_fc", f"{list(F.shape)} @ {list(P['fW'].shape)} -> ReLU -> {list(Af.shape)}", None, None))
            parts.append(Af)
        h = np.concatenate(parts, axis=1)
        mask = None
        if train and self.dropout > 0:                                         # inverted dropout
            mask = ((rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)).astype(self.dtype)
            h = h * mask
        Z = h @ P["oW"] + P["ob"]
        Pr = softmax(Z)
        tr.append(("concat_out", f"concat -> {list(h.shape)} @ {list(P['oW'].shape)} + {list(P['ob'].shape)} -> {list(Z.shape)}", None, None))
        c["out"] = dict(h=h, mask=mask, Z=Z, P=Pr)
        c["trace"] = tr
        return Pr, c

    def logits(self, Ximg, F, chunk=256):
        outs = [self.forward(Ximg[s:s + chunk], F[s:s + chunk])[1]["out"]["Z"] for s in range(0, len(F), chunk)]
        return np.concatenate(outs, axis=0)

    def predict_proba(self, Ximg, F, T=1.0, chunk=256):
        return softmax(self.logits(Ximg, F, chunk).astype(np.float64) / T)

    # ------------------------------------------------------------ Loss
    def loss(self, Pr, y):
        n = len(y)
        ce = -np.log(Pr[np.arange(n), y].astype(np.float64) + 1e-12).mean()
        reg = 0.5 * self.l2 * sum(float((w.astype(np.float64) ** 2).sum()) for k, w in self.P.items() if k[1] == "W")
        return float(ce + reg)

    # ------------------------------------------------------------ Backward
    def backward(self, y, cache):
        P, g, n = self.P, {}, len(y)
        o = cache["out"]
        dZ = o["P"].copy()
        dZ[np.arange(n), y] -= 1.0                                              # P - Y
        dZ /= n
        g["oW"] = o["h"].T @ dZ + self.l2 * P["oW"]
        g["ob"] = dZ.sum(axis=0, keepdims=True)
        dh = dZ @ P["oW"].T
        if o["mask"] is not None:
            dh = dh * o["mask"]
        col = 0
        if self.use_cnn:
            w = P["iW"].shape[1]
            dAi, col = dh[:, :w], w
            ci = cache["img"]
            dZi = dAi * (ci["Z"] > 0)
            g["iW"] = ci["flat"].T @ dZi + self.l2 * P["iW"]
            g["ib"] = dZi.sum(axis=0, keepdims=True)
            da = (dZi @ P["iW"].T).reshape(ci["a_shape"])
            for l in range(len(self.ch) - 1, -1, -1):
                cl = cache["conv"][l]
                dA = maxpool_backward(da, cl["mask"])
                dZc = dA * (cl["Z"] > 0)
                da, gW, gb = conv3x3_backward(dZc, cl["cols"], P[f"cW{l}"], cl["in_shape"])
                g[f"cW{l}"], g[f"cb{l}"] = gW + self.l2 * P[f"cW{l}"], gb
        if self.use_feat:
            dAf = dh[:, col:]
            cf = cache["feat"]
            dZf = dAf * (cf["Z"] > 0)
            g["fW"] = cf["F"].T @ dZf + self.l2 * P["fW"]
            g["fb"] = dZf.sum(axis=0, keepdims=True)
        return g

    # ------------------------------------------------------------ Adam
    def step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        if self._adam is None:
            self._adam = dict(t=0, m={k: np.zeros_like(v) for k, v in self.P.items()},
                              v={k: np.zeros_like(v) for k, v in self.P.items()})
        a = self._adam
        a["t"] += 1
        c1, c2 = 1 - b1 ** a["t"], 1 - b2 ** a["t"]
        for k in self.P:
            a["m"][k] = b1 * a["m"][k] + (1 - b1) * g[k]
            a["v"][k] = b2 * a["v"][k] + (1 - b2) * g[k] ** 2
            self.P[k] -= (lr * (a["m"][k] / c1) / (np.sqrt(a["v"][k] / c2) + eps)).astype(self.dtype)

    # ------------------------------------------------------------ Save / Load
    def state(self):
        d = {"arch": np.array("hybrid"), "cfg": np.array(json.dumps(self.cfg))}
        d.update({f"P_{k}": v for k, v in self.P.items()})
        return d

    @classmethod
    def from_state(cls, d, dtype=np.float32):
        cfg = json.loads(str(d["cfg"]))
        net = cls(**cfg, dtype=dtype)
        net.P = {k: np.array(d[f"P_{k}"], dtype=dtype) for k in net.P}
        return net

    def describe(self):
        c = self.cfg
        return dict(cnn_channels=c["ch"] if c["use_cnn"] else None, cnn_fc=c["fc_img"] if c["use_cnn"] else None,
                    feature_fc=c["fc_feat"] if c["use_feat"] else None, n_feat=c["n_feat"], concat_width=self.width,
                    n_classes=c["n_classes"], n_params=int(sum(v.size for v in self.P.values())))


def gradient_check(seeds=(0, 1), eps=1e-5):
    """เทียบ backprop ของโมเดลไฮบริด (conv + maxpool + 2 สาขา + concat) กับอนุพันธ์เชิงตัวเลข (central difference)
    ใช้ float64 และไบแอสสุ่มเพื่อเลี่ยงจุดหักของ ReLU ; คืนค่าคลาดเคลื่อนสัมพัทธ์ที่แย่ที่สุด"""
    worst = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        net = HybridNet(n_feat=6, n_classes=3, img_size=8, ch=(3, 4), fc_img=5, fc_feat=4, l2=1e-2, seed=seed, dtype=np.float64)
        for k in net.P:
            if k[1] == "b":
                net.P[k] = rng.normal(0, 0.1, net.P[k].shape)
        X, F, y = rng.normal(size=(5, 8, 8, 3)), rng.normal(size=(5, 6)), rng.integers(0, 3, 5)
        Pr, cache = net.forward(X, F)
        g = net.backward(y, cache)
        for k, p in net.P.items():
            num = np.zeros_like(p)
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + eps; lp = net.loss(net.forward(X, F)[0], y)
                p[idx] = old - eps; lm = net.loss(net.forward(X, F)[0], y)
                p[idx] = old
                num[idx] = (lp - lm) / (2 * eps)
            err = np.linalg.norm(num - g[k]) / (np.linalg.norm(num) + np.linalg.norm(g[k]) + 1e-12)
            worst = max(worst, float(err))
    return worst


if __name__ == "__main__":
    print("gradient check (hybrid CNN) worst relative error:", gradient_check())
