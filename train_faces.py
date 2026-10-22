import sqlite3
from pathlib import Path
import cv2
import json
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "geris.db"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "geris_face_model.yml"
LABELS_PATH = MODEL_DIR / "face_labels.json"


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# FACE SETTINGS
# ============================================================

FACE_SIZE = (200, 200)


# ============================================================
# GET ALL FACE SAMPLES
# ============================================================

def get_face_samples():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    samples = conn.execute("""
        SELECT
            face_samples.person_id,
            face_samples.image_path,
            persons.name
        FROM face_samples
        JOIN persons
        ON face_samples.person_id = persons.id
        ORDER BY face_samples.person_id,
                 face_samples.id
    """).fetchall()

    conn.close()

    return samples


# ============================================================
# PREPARE FACE
# ============================================================

def prepare_face(face):

    # Resize every face to the same size
    face = cv2.resize(
        face,
        FACE_SIZE
    )

    # Improve lighting consistency
    face = cv2.equalizeHist(face)

    return face


# ============================================================
# TRAIN MODEL
# ============================================================

def train_faces():

    samples = get_face_samples()

    if not samples:

        print(
            "No face samples found."
        )

        return


    training_faces = []

    labels = []

    label_names = {}


    print()
    print(
        "=============================="
    )

    print(
        "GERIS FACE TRAINING"
    )

    print(
        "=============================="
    )

    print()


    for sample in samples:

        person_id = sample["person_id"]

        person_name = sample["name"]

        image_path = (
            BASE_DIR
            / "static"
            / sample["image_path"]
        )


        print(
            f"Processing: {person_name}"
        )

        print(
            f"Image: {image_path}"
        )


        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            print(
                "  ERROR: Could not read image."
            )

            print()

            continue


        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # DETECT FACE
        # ----------------------------------------------------

        detected_faces = (
            face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )
        )


        if len(detected_faces) == 0:

            print(
                "  WARNING: No face detected."
            )

            print()

            continue


        # ----------------------------------------------------
        # LARGEST FACE
        # ----------------------------------------------------

        x, y, w, h = max(
            detected_faces,
            key=lambda face:
                face[2] * face[3]
        )


        face = gray[
            y:y + h,
            x:x + w
        ]


        # ----------------------------------------------------
        # STANDARDIZE FACE
        # ----------------------------------------------------

        face = prepare_face(
            face
        )


        # ----------------------------------------------------
        # ADD TO TRAINING DATA
        # ----------------------------------------------------

        training_faces.append(
            face
        )

        labels.append(
            person_id
        )


        label_names[
            str(person_id)
        ] = person_name


        print(
            "  Face sample added."
        )

        print()


    # ========================================================
    # CHECK TRAINING DATA
    # ========================================================

    if not training_faces:

        print(
            "No usable face samples found."
        )

        return


    unique_people = len(
        set(labels)
    )


    if unique_people < 2:

        print(
            "At least two different people "
            "are required for training."
        )

        return


    # ========================================================
    # CREATE LBPH MODEL
    # ========================================================

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=16,
            grid_x=8,
            grid_y=8
        )
    )


    labels_array = np.array(
        labels,
        dtype=np.int32
    )


    # ========================================================
    # TRAIN
    # ========================================================

    recognizer.train(
        training_faces,
        labels_array
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    recognizer.write(
        str(MODEL_PATH)
    )


    # ========================================================
    # SAVE LABELS
    # ========================================================

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_names,
            file,
            indent=4
        )


    # ========================================================
    # RESULT
    # ========================================================

    print()

    print(
        "=============================="
    )

    print(
        "GERIS FACE TRAINING COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Face samples trained: "
        f"{len(training_faces)}"
    )

    print(
        f"People trained: "
        f"{len(label_names)}"
    )

    print(
        f"Face size: "
        f"{FACE_SIZE[0]} x {FACE_SIZE[1]}"
    )

    print(
        f"Model saved: "
        f"{MODEL_PATH}"
    )

    print(
        f"Labels saved: "
        f"{LABELS_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_faces()