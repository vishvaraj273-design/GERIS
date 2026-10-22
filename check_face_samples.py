import sqlite3
import cv2

conn = sqlite3.connect("geris.db")

rows = conn.execute(
    """
    SELECT fs.image_path, p.name
    FROM face_samples fs
    JOIN persons p ON p.id = fs.person_id
    """
).fetchall()

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("=" * 50)
print("GERIS FACE SAMPLE CHECK")
print("=" * 50)

for path, name in rows:

    image = cv2.imread("static/" + path)

    if image is None:
        print(f"{name}: IMAGE NOT FOUND")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    print(f"{name}: faces detected = {len(faces)}")

conn.close()