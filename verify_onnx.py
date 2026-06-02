"""verify_onnx.py — เทียบ keypoints จาก .pt (ultralytics) กับ ONNX (onnxruntime)
เพื่อยืนยันสูตร letterbox + decode ก่อนพอร์ตเป็น JS
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "pl"))
import numpy as np, cv2
from pathlib import Path
import onnxruntime as ort
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
IMG = BASE/"dataset_v"/"val"/"images"/"100-26.jpg"
ONNX = BASE/"web"/"models"/"yolo26n_pose.onnx"
PT = BASE/"runs"/"yolo26n_pose"/"weights"/"best.pt"
S = 640

def letterbox(img, s=640):
    h, w = img.shape[:2]
    r = min(s/w, s/h)
    nw, nh = round(w*r), round(h*r)
    dw, dh = (s-nw)/2, (s-nh)/2
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh-0.1), round(dh+0.1)
    left, right = round(dw-0.1), round(dw+0.1)
    out = cv2.copyMakeBorder(rs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return out, r, left, top

def run_onnx(img, conf=0.5):
    lb, r, padx, pady = letterbox(img, S)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = np.transpose(x, (2,0,1))[None]
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # (1,300,18)
    out = out[0]
    dets = []
    for row in out:
        c = float(row[4])
        if c < conf: continue
        cls = int(round(row[5]))
        kp = []
        for k in range(4):
            kx = (row[6+k*3]   - padx)/r
            ky = (row[6+k*3+1] - pady)/r
            kp.append((kx, ky))
        cx = ((row[0]+row[2])/2 - padx)/r
        dets.append({"cls": cls, "conf": c, "cx": cx, "kp": kp})
    dets.sort(key=lambda d: d["cx"])
    return dets

def run_pt(img):
    m = YOLO(str(PT))
    r = m(img, verbose=False)[0]
    kp = r.keypoints.xy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
    bb = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    dets = [{"cls": int(cl[j]), "conf": float(cf[j]),
             "cx": (bb[j][0]+bb[j][2])/2, "kp": [tuple(kp[j][i]) for i in range(4)]}
            for j in range(len(cl))]
    dets.sort(key=lambda d: d["cx"])
    return dets

img = cv2.imdecode(np.frombuffer(open(str(IMG),'rb').read(), np.uint8), cv2.IMREAD_COLOR)
print(f"image {IMG.name}  shape={img.shape}")
po = run_pt(img); on = run_onnx(img)
print(f"pt cols={len(po)}  onnx cols={len(on)}")
labels = ["tl","tr","bl","br"]
for i,(p,o) in enumerate(zip(po,on)):
    print(f"\ncol{i}: pt cls={p['cls']} conf={p['conf']:.3f} | onnx cls={o['cls']} conf={o['conf']:.3f}")
    for k in range(4):
        dx = p['kp'][k][0]-o['kp'][k][0]; dy = p['kp'][k][1]-o['kp'][k][1]
        print(f"   {labels[k]}: pt=({p['kp'][k][0]:.1f},{p['kp'][k][1]:.1f}) "
              f"onnx=({o['kp'][k][0]:.1f},{o['kp'][k][1]:.1f})  d=({dx:+.1f},{dy:+.1f})")
