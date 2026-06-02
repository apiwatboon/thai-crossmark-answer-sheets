"""export_onnx.py — แปลง pose model (.pt) -> ONNX สำหรับรันบนเว็บด้วย onnxruntime-web

ใช้ onnx ที่ติดตั้งแยกไว้ที่ C:\\pl (ผนวกท้าย sys.path กัน numpy ชนของระบบ)
  python web/export_onnx.py                 # export yolo26n_pose (ค่าตั้งต้น)
  python web/export_onnx.py yolo11s_pose    # เลือกรุ่นอื่น
"""
import sys, shutil
# onnx ติดตั้งแยกไว้ที่ pl/ (เลี่ยงปัญหา long-path ของ site-packages) — ใส่หน้าสุดให้ชนะ
# onnx ที่ระบบ auto-install พังเสมอ; numpy ถูกลบออกจาก pl/ แล้วเพื่อไม่บัง numpy ระบบ
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "pl"))
import onnx  # noqa: F401  ยืนยันว่า onnx ใช้งานได้ก่อน ultralytics จะ auto-install ตัวที่พัง
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)

name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n_pose"
pt = BASE / "runs" / name / "weights" / "best.pt"
if not pt.exists():
    sys.exit(f"ไม่พบ weights: {pt}")

print(f"export {name}  imgsz=640  simplify=False")
m = YOLO(str(pt))
path = m.export(format="onnx", imgsz=640, simplify=False, opset=12, dynamic=False)
dst = OUT / f"{name}.onnx"
shutil.copy(str(path), str(dst))
print(f"saved -> {dst}")
print("names:", m.names)
