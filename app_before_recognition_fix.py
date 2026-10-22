from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

import sqlite3
from pathlib import Path
import base64
import cv2
import json
import numpy as np

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# BASIC CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "geris.db"

app = Flask(__name__)

app.secret_key = "geris-development-key-change-later"


# ============================================================
# OPENCV FACE DETECTION
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# FACE RECOGNITION MODEL
# ============================================================

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "geris_face_model.yml"
LABELS_PATH = MODEL_DIR / "face_labels.json"

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

if MODEL_PATH.exists():
    face_recognizer.read(str(MODEL_PATH))

if LABELS_PATH.exists():
    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        face_labels = json.load(file)
else:
    face_labels = {}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():

    conn = get_db()

    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password_hash TEXT NOT NULL,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


    # --------------------------------------------------------
    # PERSONS TABLE
    # --------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            phone TEXT,

            emergency_contact TEXT,

            face_image TEXT,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


    # --------------------------------------------------------
    # EMERGENCIES TABLE
    # --------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS emergencies (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            person_name TEXT NOT NULL,

            contact_number TEXT,

            emergency_type TEXT NOT NULL,

            location TEXT,

            description TEXT,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


    # --------------------------------------------------------
    # IDENTIFICATIONS TABLE
    # --------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS identifications (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            person_name TEXT NOT NULL,

            confidence REAL,

            identified_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


    # --------------------------------------------------------
    # FACE SAMPLES TABLE
    # --------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            person_id INTEGER NOT NULL,

            image_path TEXT NOT NULL,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (person_id)
            REFERENCES persons(id)
        )
    """)

    conn.commit()


    # --------------------------------------------------------
    # DEFAULT DEMO ACCOUNT
    # --------------------------------------------------------

    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        ("abcd",)
    ).fetchone()


    if existing is None:

        conn.execute(
            """
            INSERT INTO users
            (username, password_hash)
            VALUES (?, ?)
            """,
            (
                "abcd",
                generate_password_hash("1234")
            )
        )

        conn.commit()


    conn.close()


# ============================================================
# LOGIN
# ============================================================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        conn = get_db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        conn.close()


        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session["username"] = user["username"]

            return redirect(
                url_for("dashboard")
            )


        flash(
            "Invalid username or password.",
            "error"
        )


    return render_template(
        "login.html"
    )


# ============================================================
# USER REGISTRATION
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm = request.form.get(
            "confirm_password",
            ""
        )


        if len(username) < 3:

            flash(
                "Username must contain at least 3 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )


        if len(password) < 4:

            flash(
                "Password must contain at least 4 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )


        if password != confirm:

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template(
                "register.html"
            )


        conn = get_db()


        try:

            conn.execute(
                """
                INSERT INTO users
                (username, password_hash)
                VALUES (?, ?)
                """,
                (
                    username,
                    generate_password_hash(password)
                )
            )

            conn.commit()


        except sqlite3.IntegrityError:

            conn.close()

            flash(
                "That username is already registered.",
                "error"
            )

            return render_template(
                "register.html"
            )


        conn.close()


        flash(
            "Registration successful. Please sign in.",
            "success"
        )

        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "username" not in session:

        return redirect(
            url_for("login")
        )


    conn = get_db()


    # Registered persons

    registered_persons = conn.execute(
        """
        SELECT COUNT(*)
        FROM persons
        """
    ).fetchone()[0]


    # Emergency count

    emergency_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM emergencies
        """
    ).fetchone()[0]


    # Last identification

    last_identification = conn.execute("""
        SELECT person_name
        FROM identifications
        ORDER BY identified_at DESC
        LIMIT 1
    """).fetchone()


    if last_identification:

        last_identified_person = (
            last_identification["person_name"]
        )

    else:

        last_identified_person = "--"


    conn.close()


    return render_template(
        "dashboard.html",

        username=session["username"],

        registered_persons=registered_persons,

        emergency_count=emergency_count,

        last_identified_person=last_identified_person
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# IDENTIFY PERSON PAGE
# ============================================================

@app.route("/identify")
def identify():

    if "username" not in session:

        return redirect(
            url_for("login")
        )


    return render_template(
        "identify.html",
        username=session["username"]
    )


# ============================================================
# EMERGENCY PAGE
# ============================================================

@app.route("/emergency")
def emergency():

    if "username" not in session:

        return redirect(
            url_for("login")
        )


    return render_template(
        "emergency.html",
        username=session["username"]
    )


# ============================================================
# RESPONDERS PAGE
# ============================================================

@app.route("/responders")
def responders():

    if "username" not in session:

        return redirect(
            url_for("login")
        )


    return render_template(
        "responders.html",
        username=session["username"]
    )


# ============================================================
# INCIDENT HISTORY
# ============================================================

@app.route("/incident-history")
def incident_history():

    if "username" not in session:

        return redirect(
            url_for("login")
        )


    conn = get_db()


    emergencies = conn.execute("""
        SELECT *
        FROM emergencies
        ORDER BY created_at DESC
    """).fetchall()


    conn.close()


    return render_template(
        "incident_history.html",

        username=session["username"],

        emergencies=emergencies
    )


# ============================================================
# REGISTER PERSON
# ============================================================

@app.route(
    "/register-person",
    methods=["GET", "POST"]
)
def register_person():

    if "username" not in session:

        return redirect(
            url_for("login")
        )


    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()


        phone = request.form.get(
            "phone",
            ""
        ).strip()


        emergency_contact = request.form.get(
            "emergency_contact",
            ""
        ).strip()


        face_image = request.form.get(
            "face_image",
            ""
        )


        # ----------------------------------------------------
        # CHECK NAME
        # ----------------------------------------------------

        if not name:

            flash(
                "Please enter the person's name.",
                "error"
            )

            return render_template(
                "register_person.html",
                username=session["username"]
            )


        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if not face_image:

            flash(
                "Please capture a face image.",
                "error"
            )

            return render_template(
                "register_person.html",
                username=session["username"]
            )


        try:

            # ------------------------------------------------
            # REMOVE DATA URL PREFIX
            # ------------------------------------------------

            image_data = face_image.split(
                ",",
                1
            )[1]


            # ------------------------------------------------
            # CONVERT BASE64 TO BYTES
            # ------------------------------------------------

            image_bytes = base64.b64decode(
                image_data
            )


            # ------------------------------------------------
            # CREATE PERSON IMAGE FOLDER
            # ------------------------------------------------

            person_dir = (
                BASE_DIR
                / "static"
                / "persons"
            )


            person_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # ------------------------------------------------
            # CREATE UNIQUE IMAGE NAME
            # ------------------------------------------------

            import uuid


            image_filename = (
                str(uuid.uuid4())
                + ".jpg"
            )


            image_path = (
                person_dir
                / image_filename
            )


            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            with open(
                image_path,
                "wb"
            ) as image_file:

                image_file.write(
                    image_bytes
                )


            # ------------------------------------------------
            # DATABASE
            # ------------------------------------------------

            conn = get_db()


            # ------------------------------------------------
            # CHECK WHETHER PERSON ALREADY EXISTS
            # ------------------------------------------------

            existing_person = conn.execute(
                """
                SELECT id
                FROM persons
                WHERE lower(name) = lower(?)
                LIMIT 1
                """,
                (name,)
            ).fetchone()


            # =================================================
            # EXISTING PERSON
            # =================================================

            if existing_person:

                person_id = existing_person["id"]


                # --------------------------------------------
                # SAVE NEW FACE SAMPLE
                # --------------------------------------------

                conn.execute(
                    """
                    INSERT INTO face_samples
                    (person_id, image_path)
                    VALUES (?, ?)
                    """,
                    (
                        person_id,
                        f"persons/{image_filename}"
                    )
                )


            # =================================================
            # NEW PERSON
            # =================================================

            else:

                cursor = conn.execute(
                    """
                    INSERT INTO persons
                    (
                        name,
                        phone,
                        emergency_contact,
                        face_image
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        name,
                        phone,
                        emergency_contact,
                        f"persons/{image_filename}"
                    )
                )


                person_id = cursor.lastrowid


                # --------------------------------------------
                # SAVE FIRST FACE SAMPLE
                # --------------------------------------------

                conn.execute(
                    """
                    INSERT INTO face_samples
                    (person_id, image_path)
                    VALUES (?, ?)
                    """,
                    (
                        person_id,
                        f"persons/{image_filename}"
                    )
                )


            # ------------------------------------------------
            # COMMIT
            # ------------------------------------------------

            conn.commit()

            conn.close()


            flash(
                "Person registered successfully.",
                "success"
            )


            return redirect(
                url_for("register_person")
            )


        except Exception as error:

            print(
                "Registration error:",
                error
            )


            flash(
                "Unable to register person.",
                "error"
            )


    return render_template(
        "register_person.html",
        username=session["username"]
    )


# ============================================================
# SAVE CAPTURE
# ============================================================

@app.route(
    "/save-capture",
    methods=["POST"]
)
def save_capture():

    if "username" not in session:

        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401


    data = request.get_json()


    if not data or "image" not in data:

        return jsonify({
            "success": False,
            "message": "No image received"
        }), 400


    try:

        image_data = data["image"]


        image_data = image_data.split(
            ",",
            1
        )[1]


        image_bytes = base64.b64decode(
            image_data
        )


        capture_dir = (
            BASE_DIR
            / "static"
            / "captures"
        )


        capture_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        image_path = (
            capture_dir
            / "captured_person.jpg"
        )


        with open(
            image_path,
            "wb"
        ) as image_file:

            image_file.write(
                image_bytes
            )


        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            return jsonify({
                "success": False,
                "message": "Unable to read captured image"
            }), 500


        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # DETECT FACES
        # ----------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # ----------------------------------------------------
        # DRAW FACE RECTANGLES
        # ----------------------------------------------------

        for (
            x,
            y,
            w,
            h
        ) in faces:

            cv2.rectangle(
                image,

                (x, y),

                (
                    x + w,
                    y + h
                ),

                (0, 255, 0),

                2
            )


        # ----------------------------------------------------
        # SAVE PROCESSED IMAGE
        # ----------------------------------------------------

        cv2.imwrite(
            str(image_path),
            image
        )


        return jsonify({
            "success": True,
            "message": "Image captured successfully",
            "face_count": len(faces),
            "filename": "captures/captured_person.jpg"
        })


    except Exception as error:

        print(
            "Save error:",
            error
        )


        return jsonify({
            "success": False,
            "message": "Unable to process image"
        }), 500


# ============================================================
# FACE RECOGNITION
# ============================================================
@app.route(
    "/recognize-face",
    methods=["POST"]
)
def recognize_face():

    if "username" not in session:

        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json()

    if not data or "image" not in data:

        return jsonify({
            "success": False,
            "message": "No image received"
        }), 400

    try:

        # ====================================================
        # DECODE IMAGE
        # ====================================================

        image_data = data["image"]

        image_data = image_data.split(
            ",",
            1
        )[1]

        image_bytes = base64.b64decode(
            image_data
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "success": False,
                "message": "Unable to read image"
            }), 400


        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # DETECT FACE
        # ====================================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        if len(faces) == 0:

            return jsonify({
                "success": False,
                "message": "No face detected"
            })


        # ====================================================
        # LARGEST FACE
        # ====================================================

        x, y, w, h = max(
            faces,
            key=lambda face:
                face[2] * face[3]
        )


        face = gray[
            y:y + h,
            x:x + w
        ]


        # ====================================================
        # SAME PROCESSING USED DURING TRAINING
        # ====================================================

        face = cv2.resize(
            face,
            (200, 200)
        )


        face = cv2.equalizeHist(
            face
        )


        # ====================================================
        # FACE RECOGNITION
        # ====================================================

        label, confidence = (
            face_recognizer.predict(
                face
            )
        )


        print()
        print("==============================")
        print("GERIS FACE RECOGNITION")
        print("==============================")
        print(
            "Predicted label:",
            label
        )
        print(
            "Confidence:",
            confidence
        )
        print(
            "Labels:",
            face_labels
        )


        # ====================================================
        # GET PERSON NAME
        # ====================================================

        person_name = face_labels.get(
            str(label),
            "Unknown Person"
        )


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================
        #
        # LBPH:
        # Lower confidence = better match
        #
        # A stricter threshold prevents weak matches
        # from being reported as real identities.
        # ====================================================

        if confidence <= 70:

            identified = True

        else:

            identified = False

            person_name = "Unknown Person"


        # ====================================================
        # SAVE SUCCESSFUL IDENTIFICATION
        # ====================================================

        if identified:

            conn = get_db()

            conn.execute(
                """
                INSERT INTO identifications
                (person_name, confidence)
                VALUES (?, ?)
                """,
                (
                    person_name,
                    confidence
                )
            )

            conn.commit()

            conn.close()


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({
            "success": True,
            "identified": identified,
            "name": person_name,
            "confidence": round(
                confidence,
                2
            )
        })


    except Exception as error:

        print(
            "Recognition error:",
            error
        )

        return jsonify({
            "success": False,
            "message": "Face recognition failed"
        }), 500

# ============================================================
# CREATE EMERGENCY
# ============================================================

@app.route(
    "/api/create-emergency",
    methods=["POST"]
)
def create_emergency():

    if "username" not in session:

        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401


    data = request.get_json()


    if not data:

        return jsonify({
            "success": False,
            "message": "No emergency data received"
        }), 400


    person_name = data.get(
        "person_name",
        ""
    ).strip()


    contact_number = data.get(
        "contact_number",
        ""
    ).strip()


    emergency_type = data.get(
        "emergency_type",
        ""
    ).strip()


    location = data.get(
        "location",
        ""
    ).strip()


    description = data.get(
        "description",
        ""
    ).strip()


    if not person_name or not emergency_type:

        return jsonify({
            "success": False,
            "message": (
                "Person name and emergency type "
                "are required"
            )
        }), 400


    try:

        conn = get_db()


        conn.execute(
            """
            INSERT INTO emergencies
            (
                person_name,
                contact_number,
                emergency_type,
                location,
                description
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person_name,
                contact_number,
                emergency_type,
                location,
                description
            )
        )


        conn.commit()

        conn.close()


        return jsonify({
            "success": True,
            "message": "Emergency saved successfully"
        })


    except Exception as error:

        print(
            "Emergency error:",
            error
        )


        return jsonify({
            "success": False,
            "message": "Failed to save emergency"
        }), 500


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True
    )