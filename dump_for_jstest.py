"""dump_for_jstest.py — รัน onnx แล้ว dump raw output + geo + ผล decode(python)
ให้ Node เอาไปทดสอบ decodeE2E/ordering ของ pipeline.js ว่าตรงกัน
+ dump ชุดทดสอบ polyfit เทียบ numpy
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "pl"))
import numpy as np, cv2
from pathlib import Path
import onnxruntime as ort

BASE = Path(__file__).resolve().parent.parent
IMG = BASE/"dataset_v"/"val"/"images"/"100-26.jpg"
ONNX = BASE/"web"/"models"/"yolo26n_pose.onnx"
S = 640

def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s/w, s/h)
    nw, nh = round(w*r), round(h*r); dw, dh = (s-nw)/2, (s-nh)/2
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(rs, round(dh-0.1), round(dh+0.1), round(dw-0.1), round(dw+0.1),
                             cv2.BORDER_CONSTANT, value=(114,114,114))
    return out, r, round(dw-0.1), round(dh-0.1)

img = cv2.imdecode(np.frombuffer(open(str(IMG),'rb').read(), np.uint8), cv2.IMREAD_COLOR)
lb, r, padx, pady = letterbox(img, S)
x = np.transpose(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0, (2,0,1))[None]
sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # (1,300,18)

# python decode (เหมือน verify) -> เรียงตาม cx
POSE_CONF = 0.30
dets = []
for row in out[0]:
    c = float(row[4])
    if c < POSE_CONF: continue
    un = lambda px, py: [float((px-padx)/r), float((py-pady)/r)]
    dets.append({"cls": int(round(row[5])), "conf": c,
                 "cx": float(((row[0]+row[2])/2-padx)/r),
                 "tl": un(row[6],row[7]), "tr": un(row[9],row[10]),
                 "bl": un(row[12],row[13]), "br": un(row[15],row[16])})
dets.sort(key=lambda d: d["cx"])

# polyfit test set
rng = np.random.default_rng(0)
xv = rng.uniform(0, 100, 25); yv = 1.7*xv + 5 + rng.normal(0, 3, 25)
pf = np.polyfit(xv, yv, 1)

json.dump({
    "out": out.reshape(-1).tolist(), "dims": list(out.shape),
    "geo": {"r": r, "padx": padx, "pady": pady}, "pose_conf": POSE_CONF,
    "py_dets": dets,
    "polyfit": {"xv": xv.tolist(), "yv": yv.tolist(), "expect": pf.tolist()},
}, open(str(BASE/"web"/"jstest_data.json"), "w"))
print("dumped web/jstest_data.json  py_cols=", len(dets))
