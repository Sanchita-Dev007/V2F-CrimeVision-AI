"""
history_lookup.py (DeepFace version, RetinaFace detector)
--------------------------------------------------------------
Given a generated suspect composite (from generate_face.py), this module
searches the mock case database (built by mock_database.py) for visually
similar faces and returns their case history.

IMPORTANT — framing for your README/viva:
This demonstrates HOW a real system would plug into an actual records
database. It is not connected to any real law-enforcement data source.
Face-matching systems also have well-documented accuracy and fairness
limitations (varying accuracy across demographics, lighting, image
quality) — mention this explicitly in your project write-up.
"""

import json
import pickle
import numpy as np
from deepface import DeepFace

DATABASE_JSON = "database.json"
EMBEDDINGS_PKL = "embeddings.pkl"
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "retinaface"

# Cosine similarity ranges from -1 to 1 (1 = identical direction/same face).
# 0.70+ is a reasonable "likely same person" starting threshold for Facenet
# embeddings — tune this after testing with your own data.
DEFAULT_THRESHOLD = 0.70


def load_database():
    """Load case metadata and their precomputed face embeddings."""
    with open(DATABASE_JSON, "r") as f:
        records = json.load(f)
    with open(EMBEDDINGS_PKL, "rb") as f:
        embeddings = pickle.load(f)
    return records, embeddings


def cosine_similarity(vec_a, vec_b):
    vec_a, vec_b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def match_face(generated_image_path, threshold=DEFAULT_THRESHOLD):
    """
    Compare a generated suspect composite against the mock case database.

    Returns a dict with status, message, and a list of matches (sorted best
    match first), each with case info + similarity score.
    """
    records, known_embeddings = load_database()

    try:
        result = DeepFace.represent(
            img_path=generated_image_path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND,
        )
        query_embedding = result[0]["embedding"]
    except ValueError:
        return {
            "status": "no_face_detected",
            "message": "No face could be detected in the generated composite.",
            "matches": [],
        }

    matches = []
    for record, known_embedding in zip(records, known_embeddings):
        similarity = cosine_similarity(query_embedding, known_embedding)
        if similarity >= threshold:
            matches.append({
                "case_id": record["case_id"],
                "crime_type": record["crime_type"],
                "fir_count": record["fir_count"],
                "status": record["status"],
                "last_reported": record["last_reported"],
                "similarity_score": round(similarity * 100, 1),
            })

    # Best match (highest similarity) first
    matches.sort(key=lambda m: m["similarity_score"], reverse=True)

    return {
        "status": "match_found" if matches else "no_match",
        "message": f"{len(matches)} potential match(es) found." if matches
                   else "No matches found in the case database.",
        "matches": matches,
    }


if __name__ == "__main__":
    # Quick manual test — point this at one of your generated outputs
    test_image = "outputs/ai_suspect.png"
    result = match_face(test_image)
    print(result["message"])
    for m in result["matches"]:
        print(f"  {m['case_id']} | {m['crime_type']} | FIRs: {m['fir_count']} "
              f"| Status: {m['status']} | Similarity: {m['similarity_score']}%")