"""api.py - REST API จำแนกสภาพอากาศจากรูปภาพ (โมเดลไฮบริด CNN + คุณลักษณะ 36 มิติ ด้วย NumPy)

รัน:   python api.py                  (พอร์ต 8000)
ใช้:   curl -F "image=@sky.jpg" http://localhost:8000/predict
       curl -F "image=@sky.jpg" "http://localhost:8000/predict?trace=1"   # แสดงมิติเมทริกซ์ทุกขั้นตอน
"""
import os
from flask import Flask, jsonify, request, send_from_directory

from weather.pipeline import Predictor

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024        # จำกัดไฟล์ 25 MB
predictor = Predictor()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", classes=predictor.classes, rain_classes=predictor.rain_classes, rain_weights=predictor.rain_weights, feature_mode=predictor.feat,
                   temperature=predictor.temp, model=predictor.net.describe())


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(status="error", message="ต้องส่งไฟล์รูปในฟิลด์ชื่อ 'image' (multipart/form-data)"), 400
    data = f.read()
    if not data:
        return jsonify(status="error", message="ไฟล์ว่างเปล่า"), 400
    try:
        out = predictor.predict(data, trace=request.args.get("trace") == "1")
    except Exception:                                          # ไฟล์ที่ไม่ใช่รูป / รูปเสีย
        return jsonify(status="error", message="อ่านไฟล์เป็นรูปภาพไม่ได้ (รองรับ JPG/PNG/WEBP/BMP)"), 400
    if out["status"] == "rejected":                            # ไม่ผ่านการตรวจสอบภาพ
        return jsonify(out), 422
    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify(status="error", message="ไฟล์ใหญ่เกิน 25 MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
