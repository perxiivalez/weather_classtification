import json
import os
import re
import numpy as np

from . import imageops as ops
from .cnn import HybridNet
from .mlp import softmax

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "weights.npz")


DEFAULT_RAIN_WEIGHTS = {"Rain": 1.0, "Storm Cloudy": 0.7}


def _norm(s):
    return re.sub(r"[\s_\-]+", "", str(s)).lower()


def rain_probability(prob, classes, weights):
    idx = {_norm(c): i for i, c in enumerate(classes)}
    parts = {}
    for name, w in weights.items():
        i = idx.get(_norm(name))
        if i is not None:
            parts[classes[i]] = dict(prob=float(prob[i]), weight=float(w), contribution=float(w) * float(prob[i]))
    if not parts:
        return None, parts
    return min(1.0, sum(v["contribution"] for v in parts.values())), parts


class Predictor:
    def __init__(self, weights_path=DEFAULT_MODEL):
        d = np.load(weights_path, allow_pickle=False)
        if "arch" not in d.files:
            raise SystemExit("weights.npz เป็นโมเดล MLP รุ่นเก่า — เทรนใหม่ด้วย `python train.py --final` เพื่อให้ได้โมเดล CNN+feature")
        self.net = HybridNet.from_state({k: d[k] for k in d.files})
        self.mean, self.std = d["mean"], d["std"]
        self.img_mean, self.img_std = d["img_mean"], d["img_std"]
        self.classes = [str(c) for c in d["classes"]]
        self.feat = str(d["feat"])
        self.temp = float(d["temp"]) if "temp" in d.files else 1.0
    
        w = dict(DEFAULT_RAIN_WEIGHTS)
        have = {_norm(k) for k in w}
        if "rain_classes" in d.files:
            for c in d["rain_classes"]:
                if _norm(c) not in have:
                    w[str(c)] = 1.0
        cfg = os.path.join(os.path.dirname(os.path.abspath(weights_path)), "rain_config.json")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8") as f:
                w.update({str(k): float(v) for k, v in json.load(f).get("rain_weights", {}).items()})
        
        idx = {_norm(c): c for c in self.classes}
        self.rain_weights = {idx[_norm(k)]: v for k, v in w.items() if _norm(k) in idx}
        self.rain_classes = list(self.rain_weights)

  
    def predict(self, source, trace=False):
        steps = []
        t = (lambda name, **kw: steps.append(dict(step=name, **kw))) if trace else (lambda *a, **k: None)

        
        img = ops.load_rgb(source)
        H, W, C = img.shape
        t("1_image_to_matrix", matrix_shape=[H, W, C], value_range=[float(img.min()), float(img.max())],
          note="สมาชิกแต่ละตัวคือความเข้มของพิกเซลต่อช่องสี R,G,B (ปรับเป็น 0..1)")

        
        P = img.reshape(-1, 3)
        b = ops.brightness(img)
        t("2a_brightness", pixel_matrix_shape=list(P.shape), weight_vector=ops.LUMA_W.tolist(),
          formula="mean(P @ w)", mean_brightness=round(b, 4))
        YCC = P @ ops.RGB2YCC.T
        t("2b_sky_detector", transform_matrix_3x3=ops.RGB2YCC.round(4).tolist(), result_shape=list(YCC.shape),
          formula="YCC = P @ T^T  แล้วจัดกลุ่มพิกเซลเทา/น้ำเงิน/ส้มเป็นท้องฟ้า", sky_ratio=round(ops.sky_ratio(img), 4))
        ok, report = ops.validate(img)
        t("2c_validation", passed=ok, **{k: v for k, v in report.items() if k != "shape"})
        if not ok:
            return dict(status="rejected", reasons=report["problems"], validation=report,
                        trace=steps if trace else None)

       
        small = ops.resize(img, ops.IMG_SIZE)
        Rh, Rw = ops.area_matrix(H, ops.IMG_SIZE), ops.area_matrix(W, ops.IMG_SIZE)
        t("3a_resize", R_h_shape=list(Rh.shape), R_w_shape=list(Rw.shape), formula="X' = R_h X R_w^T (ต่อช่องสี)",
          result_shape=list(small.shape), row_sum_check=float(Rh.sum(1).mean()))
        F = ops.build_features(small[None], self.feat)          
        x = ops.normalize(F, self.mean, self.std)
        t("3b_vectorize_normalize", feature_mode=self.feat, vector_shape=list(x.shape),
          formula="x' = (x - mu) / sigma", mean_after=round(float(x.mean()), 4), std_after=round(float(x.std()), 4))
        xi = ((small - self.img_mean) / self.img_std)[None]      
        t("3c_cnn_input", shape=list(xi.shape), formula="X' = (X - mu_rgb) / sigma_rgb")

        
        _, cache = self.net.forward(xi, x)
        for name, desc, active, total in cache["trace"]:
            extra = {} if active is None else dict(active_units=active, total_units=total)
            t(f"4_{name}", layer=desc, **extra)
        logits = cache["out"]["Z"].astype(np.float64)


        prob = softmax(logits / self.temp)[0]
        t("5_softmax", logits=logits[0].round(4).tolist(), temperature=self.temp,
          formula="p_k = exp(z_k/T) / sum_j exp(z_j/T)", probabilities=prob.round(4).tolist(), sum=float(prob.sum()))


        k = int(prob.argmax())
        rain_p, rain_parts = rain_probability(prob, self.classes, self.rain_weights)
        out = dict(status="ok", predicted_class=self.classes[k], confidence=round(float(prob[k]), 4),
                   confidence_percent=round(100 * float(prob[k]), 1),
                   rain_probability_percent=None if rain_p is None else round(100 * rain_p, 1),
                   rain_breakdown={c: dict(prob_percent=round(100 * v["prob"], 1), weight=v["weight"],
                                           contribution_percent=round(100 * v["contribution"], 1))
                                   for c, v in rain_parts.items()},
                   probabilities={c: round(100 * float(p), 1) for c, p in zip(self.classes, prob)},
                   validation=report)
        t("6_result", predicted_index=k, rain_weights=self.rain_weights, formula="rain = min(1, sum_k w_k * p_k)",
          **{kk: out[kk] for kk in ("predicted_class", "confidence_percent", "rain_probability_percent", "rain_breakdown")})
        if trace:
            out["trace"] = steps
        return out
