"""export_bbox.py — แปลง bbox detector (.pt) จาก runs_cross -> ONNX สำหรับ ensemble บนเว็บ
ใช้ onnx ที่ติดตั้งแยกไว้ที่ pl/ (เลี่ยง long-path) เหมือน export_onnx.py
  python web/export_bbox.py            # export yolo26n, yolo11s, yolov8s (ค่าตั้งต้น)
"""
import sys, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "pl"))
import onnx  # noqa: F401  ยืนยัน onnx ใช้ได้ก่อน ultralytics auto-install ตัวพัง
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)

names = sys.argv[1:] or ["yolo26n", "yolo11s", "yolov8s"]
for name in names:
    pt = BASE / "runs_cross" / name / "weights" / "best.pt"
    if not pt.exists():
        print(f"ไม่พบ weights: {pt} — ข้าม"); continue
    print(f"export {name}_bbox  imgsz=640")
    m = YOLO(str(pt))
    path = m.export(format="onnx", imgsz=640, simplify=False, opset=12, dynamic=False)
    dst = OUT / f"{name}_bbox.onnx"
    shutil.copy(str(path), str(dst))
    print(f"saved -> {dst}  names={m.names}")
