"""
mock_database.py (DeepFace version, RetinaFace detector)
-----------------------------------------------------------
Builds a SIMULATED case-record database for CrimeVision-AI's history-correlation
module. This is NOT connected to any real police system — it exists to
demonstrate, end-to-end, how a real integration would work.

Uses DeepFace (TensorFlow-based, no C++ compiler needed) with the RetinaFace
detector backend, which handles stylized/AI-generated images better than
DeepFace's default detector.

Install:
    pip install deepface tf-keras

First run will auto-download model weights (~90MB Facenet + RetinaFace
weights) — this is normal and only happens once.
"""

import os
import json
import pickle
from deepface import DeepFace

# ---- Configuration ----
CASE_IMAGES_DIR = "mock_case_photos"
DATABASE_JSON = "database.json"
EMBEDDINGS_PKL = "embeddings.pkl"
MODEL_NAME = "Facenet"          # good accuracy/speed balance
DETECTOR_BACKEND = "retinaface"  # more robust than default for stylized images

# ---- Synthetic case metadata ----
# Fictional data, purely to demonstrate the pipeline end-to-end.
MOCK_CASES = [
    {
        "case_id": "FIR-2023-00142",
        "image_file": "case_1.jpg",
        "crime_type": "Theft",
        "fir_count": 2,
        "status": "Under Investigation",
        "last_reported": "2023-08-14",
    },
    {
        "case_id": "FIR-2022-00098",
        "image_file": "case_2.jpg",
        "crime_type": "Assault",
        "fir_count": 1,
        "status": "Closed",
        "last_reported": "2022-11-02",
    },
    {
        "case_id": "FIR-2024-00211",
        "image_file": "case_3.jpg",
        "crime_type": "Fraud",
        "fir_count": 3,
        "status": "Under Investigation",
        "last_reported": "2024-03-19",
    },
    # Add more mock entries as needed, with a matching image in CASE_IMAGES_DIR.
]


def build_database():
    """Compute face embeddings for each mock case image and save the database."""
    records = []
    embeddings = []

    for case in MOCK_CASES:
        image_path = os.path.join(CASE_IMAGES_DIR, case["image_file"])
        if not os.path.exists(image_path):
            print(f"[WARNING] Missing image for {case['case_id']}: {image_path} — skipping.")
            continue

        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                enforce_detection=True,
                detector_backend=DETECTOR_BACKEND,
            )
            embedding = result[0]["embedding"]
        except ValueError:
            print(f"[WARNING] No face detected in {image_path} — skipping.")
            continue

        embeddings.append(embedding)
        records.append(case)
        print(f"[OK] Encoded {case['case_id']}")

    with open(DATABASE_JSON, "w") as f:
        json.dump(records, f, indent=2)

    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\nDatabase built: {len(records)} records saved.")
    print(f"  -> {DATABASE_JSON}")
    print(f"  -> {EMBEDDINGS_PKL}")


if __name__ == "__main__":
    if not os.path.exists(CASE_IMAGES_DIR):
        os.makedirs(CASE_IMAGES_DIR)
        print(f"Created '{CASE_IMAGES_DIR}/' — add your mock case face images there "
              f"(case_1.jpg, case_2.jpg, case_3.jpg per MOCK_CASES above), then rerun this script.")
    else:
        build_database()