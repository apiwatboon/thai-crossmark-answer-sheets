"""py_ref.py — รัน desktop pipeline (grader_core) ด้วยหลายรุ่น pose -> dump คำตอบเป็น JSON
ground-truth เทียบกับผลฝั่ง JS (เบราว์เซอร์). crop=grid (expand+crop), reader=pixel_count
"""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grader_core as gc

BASE = Path(__file__).resolve().parent.parent
SHEETS = [
    BASE/"dataset_v"/"val"/"images"/"100-26.jpg",
    BASE/"dataset_v"/"val"/"images"/"100-27.jpg",
    BASE/"dataset_v"/"val"/"images"/"100-32.jpg",
    BASE/"dataset_v"/"val_easy"/"images"/"100-25.jpg",
    BASE/"dataset_v"/"val_easy"/"images"/"100-31.jpg",
]
MODELS = ["yolo26n_pose", "yolo26s_pose", "yolo11n_pose", "yolo11s_pose", "yolov8n_pose", "yolov8s_pose"]

for sp in SHEETS:
    if sp.exists():
        shutil.copy(str(sp), str(BASE/"web"/("t_"+sp.name)))

ref = {}
for model in MODELS:
    grader = gc.Grader(pose=model, reader="pixel_count", crop="grid")
    ref[model] = {}
    for sp in SHEETS:
        if not sp.exists():
            continue
        img = gc.rp.read_image(str(sp))
        results, cols_meta = grader.grade(img)
        ref[model][sp.name] = [{"question": r["question"], "answer": r["answer"]} for r in results]
        print(f"{model} {sp.name}: {len(results)} answers, cols={len(cols_meta)}")
json.dump(ref, open(str(BASE/"web"/"py_ref.json"), "w"))
print("dumped py_ref.json")
