import json
import sys

from weather.pipeline import Predictor

if len(sys.argv) < 2:
    sys.exit("ใช้: python demo_trace.py <ไฟล์รูป>")

out = Predictor().predict(sys.argv[1], trace=True)
for s in out["trace"]:
    name = s.pop("step")
    print(f"\n[{name}]")
    for k, v in s.items():
        print(f"   {k}: {json.dumps(v, ensure_ascii=False)}")
print("\n==== ผลลัพธ์ ====")
print(json.dumps({k: v for k, v in out.items() if k != "trace"}, ensure_ascii=False, indent=2))
