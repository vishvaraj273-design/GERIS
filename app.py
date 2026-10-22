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
import math
import numpy as np
import secrets
import time

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from insightface.app import FaceAnalysis


# ============================================================
# BASIC CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "geris.db"

app = Flask(__name__)

app.secret_key = "geris-development-key-change-later"

RESPONDER_LOCATION_MAX_AGE_SECONDS = 5 * 60
EMERGENCY_RESPONDER_TYPES = {
    "Cardiac Arrest": ("Ambulance", "Medical"),
    "Medical Emergency": ("Ambulance", "Medical"),
    "Fire": ("Fire & Rescue",),
    "Accident": ("Ambulance", "Police"),
    "Other": ()
}
ACTIVE_ASSIGNMENT_STATUSES = (
    "Assigned",
    "Accepted",
    "On the way",
    "Arrived"
)
ASSIGNMENT_STATUS_TRANSITIONS = {
    "Assigned": {"Accepted", "Declined"},
    "Accepted": {"On the way"},
    "On the way": {"Arrived"},
    "Arrived": {"Resolved"}
}


# ============================================================
# FACE DETECTION / DEEP LEARNING
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_analysis = FaceAnalysis(name="buffalo_l")
face_analysis.prepare(ctx_id=-1, det_size=(640, 640))

IDENTITY_THRESHOLD = 0.78
MINIMUM_MATCH_MARGIN = 0.08


# ============================================================
# FACE PREPROCESSING
# ============================================================

def analyze_primary_face(image):

    detections = face_analysis.get(image)
    resized_with_padding = False
    padding = 0

    if not detections:
        height, width = image.shape[:2]
        if max(height, width) > 400:
            return None

        padding = max(32, int(max(height, width) * 0.5))
        padded_image = cv2.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_REPLICATE
        )
        analysis_image = cv2.resize(
            padded_image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )
        detections = face_analysis.get(analysis_image)
        resized_with_padding = True

    if not detections:
        return None

    detected_face = max(
        detections,
        key=lambda face: (
            face.bbox[2] - face.bbox[0]
        ) * (
            face.bbox[3] - face.bbox[1]
        )
    )

    height, width = image.shape[:2]
    coordinates = np.asarray(
        detected_face.bbox,
        dtype=np.float32
    )
    if resized_with_padding:
        coordinates = coordinates / 2 - padding

    x1, y1, x2, y2 = (
        int(round(value))
        for value in coordinates
    )
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    embedding = np.asarray(
        detected_face.embedding,
        dtype=np.float32
    )
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None

    return {
        "crop": cv2.resize(face_crop, (256, 256)),
        "embedding": embedding / norm
    }


def serialize_embedding(embedding):
    return json.dumps(
        np.asarray(embedding, dtype=np.float32).tolist()
    )


def deserialize_embedding(value):
    if not value:
        return None

    try:
        embedding = np.asarray(json.loads(value), dtype=np.float32)
    except (TypeError, ValueError):
        return None

    norm = np.linalg.norm(embedding)
    if embedding.size == 0 or norm == 0:
        return None

    return embedding / norm


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(DB_PATH, timeout=30)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def validate_location(data):

    values = []
    for key in ("latitude", "longitude", "accuracy"):
        raw_value = data.get(key)
        if isinstance(raw_value, bool):
            return None, f"Invalid {key} value."
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None, f"Invalid {key} value."

        if not math.isfinite(value):
            return None, f"Invalid {key} value."
        values.append(value)

    latitude, longitude, accuracy = values
    if not -90 <= latitude <= 90:
        return None, "Latitude must be between -90 and 90."
    if not -180 <= longitude <= 180:
        return None, "Longitude must be between -180 and 180."
    if accuracy < 0:
        return None, "Accuracy must be zero or greater."

    return (latitude, longitude, accuracy), None


def validate_latitude_longitude(latitude, longitude):
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        return None, "Latitude and longitude must be numeric values."

    if not math.isfinite(latitude_value):
        return None, "Latitude must be a finite number."
    if not math.isfinite(longitude_value):
        return None, "Longitude must be a finite number."
    if not -90 <= latitude_value <= 90:
        return None, "Latitude must be between -90 and 90."
    if not -180 <= longitude_value <= 180:
        return None, "Longitude must be between -180 and 180."

    return (latitude_value, longitude_value), None


def normalize_responder_status(value):
    if value is None:
        return None, "Availability status is required."

    status = str(value).strip()
    if not status:
        return None, "Availability status is required."

    allowed_statuses = {
        "available": "Available",
        "busy": "Busy",
        "offline": "Offline"
    }

    normalized = status.lower()
    if normalized not in allowed_statuses:
        return None, "Availability status must be Available, Busy, or Offline."

    return allowed_statuses[normalized], None


def distance_meters(latitude_a, longitude_a, latitude_b, longitude_b):

    earth_radius = 6371000
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(math.radians(latitude_a))
        * math.cos(math.radians(latitude_b))
        * math.sin(longitude_delta / 2) ** 2
    )

    return earth_radius * 2 * math.atan2(
        math.sqrt(haversine),
        math.sqrt(1 - haversine)
    )


def get_active_emergencies(conn, owner_username=None):

    query = """
        SELECT
            id,
            person_name,
            contact_number,
            emergency_type,
            description,
            latitude,
            longitude,
            accuracy,
            status,
            tracking_active,
            created_at,
            updated_at
        FROM emergencies
        WHERE status = 'active'
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
    """
    params = ()

    if owner_username is not None:
        query += " AND owner_username = ?"
        params = (owner_username,)

    query += " ORDER BY created_at DESC"

    return conn.execute(query, params).fetchall()


def add_assignment_history(
    conn,
    assignment_id,
    previous_status,
    new_status,
    actor_role,
    actor_responder_id=None,
    actor_username=None
):
    conn.execute(
        """
        INSERT INTO assignment_history (
            assignment_id,
            previous_status,
            new_status,
            actor_role,
            actor_responder_id,
            actor_username
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            assignment_id,
            previous_status,
            new_status,
            actor_role,
            actor_responder_id,
            actor_username
        )
    )


def try_assign_emergency(conn, incident_id, now=None):
    now = time.time() if now is None else now
    active_placeholders = ", ".join("?" for _ in ACTIVE_ASSIGNMENT_STATUSES)
    existing = conn.execute(
        f"""
        SELECT id, status
        FROM emergency_assignments
        WHERE emergency_id = ?
          AND status IN ({active_placeholders})
        ORDER BY assigned_at DESC
        LIMIT 1
        """,
        (incident_id, *ACTIVE_ASSIGNMENT_STATUSES)
    ).fetchone()
    if existing is not None:
        return {
            "status": "already_assigned",
            "assignment_id": existing["id"],
            "assignment_status": existing["status"]
        }

    incident = conn.execute(
        """
        SELECT emergency_type, latitude, longitude, status
        FROM emergencies
        WHERE id = ?
        """,
        (incident_id,)
    ).fetchone()
    if incident is None or incident["status"] != "active":
        return {"status": "incident_unavailable"}

    incident_coords, coord_error = validate_latitude_longitude(
        incident["latitude"],
        incident["longitude"]
    )
    if coord_error:
        return {"status": "no_eligible_responder"}

    suitable_types = EMERGENCY_RESPONDER_TYPES.get(
        incident["emergency_type"],
        ()
    )
    if not suitable_types:
        return {"status": "no_eligible_responder"}

    type_placeholders = ", ".join("?" for _ in suitable_types)
    responders = conn.execute(
        f"""
        SELECT id, responder_type, latitude, longitude, location_updated_at
        FROM responders
        WHERE availability_status = 'Available'
          AND responder_type IN ({type_placeholders})
          AND location_updated_at >= ?
          AND location_updated_at <= ?
        """,
        (*suitable_types, now - RESPONDER_LOCATION_MAX_AGE_SECONDS, now)
    ).fetchall()

    candidates = []
    for responder in responders:
        responder_coords, responder_coord_error = validate_latitude_longitude(
            responder["latitude"],
            responder["longitude"]
        )
        if responder_coord_error:
            continue
        candidates.append(
            (
                distance_meters(
                    incident_coords[0],
                    incident_coords[1],
                    responder_coords[0],
                    responder_coords[1]
                ),
                responder["id"]
            )
        )

    if not candidates:
        return {"status": "no_eligible_responder"}

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    _, responder_id = candidates[0]

    cursor = conn.execute(
        """
        INSERT INTO emergency_assignments (emergency_id, responder_id, status)
        VALUES (?, ?, 'Assigned')
        """,
        (incident_id, responder_id)
    )
    assignment_id = cursor.lastrowid
    responder_update = conn.execute(
        """
        UPDATE responders
        SET availability_status = 'Busy',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND availability_status = 'Available'
        """,
        (responder_id,)
    )
    if responder_update.rowcount != 1:
        raise sqlite3.IntegrityError(
            "Selected responder is no longer available."
        )

    add_assignment_history(
        conn,
        assignment_id,
        None,
        "Assigned",
        "system"
    )
    return {
        "status": "assigned",
        "assignment_id": assignment_id,
        "assignment_status": "Assigned"
    }


def ensure_face_embedding_schema():

    conn = get_db()

    columns = conn.execute(
        "PRAGMA table_info(face_samples)"
    ).fetchall()

    if columns and not any(
        column[1] == "embedding"
        for column in columns
    ):
        conn.execute(
            "ALTER TABLE face_samples ADD COLUMN embedding TEXT"
        )
        conn.commit()

    conn.close()


def load_registered_embeddings():

    conn = get_db()
    samples = conn.execute("""
        SELECT
            face_samples.id AS sample_id,
            face_samples.person_id,
            face_samples.image_path,
            face_samples.embedding,
            persons.name,
            persons.phone,
            persons.emergency_contact
        FROM face_samples
        JOIN persons
        ON face_samples.person_id = persons.id
        ORDER BY face_samples.person_id, face_samples.id
    """).fetchall()

    registered = []
    updates = []

    for sample in samples:
        embedding = deserialize_embedding(sample["embedding"])
        if embedding is None:
            image_path = BASE_DIR / "static" / sample["image_path"]
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            face_data = analyze_primary_face(image)
            if face_data is None:
                continue

            embedding = face_data["embedding"]
            updates.append((
                serialize_embedding(embedding),
                sample["sample_id"]
            ))

        registered.append({
            "person_id": sample["person_id"],
            "name": sample["name"],
            "phone": sample["phone"],
            "emergency_contact": sample["emergency_contact"],
            "embedding": embedding
        })

    if updates:
        conn.executemany(
            "UPDATE face_samples SET embedding = ? WHERE id = ?",
            updates
        )
        conn.commit()

    conn.close()
    return registered


def match_registered_person(query_embedding, registered_samples):

    by_person = {}

    for sample in registered_samples:
        score = float(np.dot(
            query_embedding,
            sample["embedding"]
        ))
        person_id = sample["person_id"]

        if person_id not in by_person:
            by_person[person_id] = {
                "person_id": person_id,
                "name": sample["name"],
                "phone": sample["phone"],
                "emergency_contact": sample["emergency_contact"],
                "scores": []
            }

        by_person[person_id]["scores"].append(score)

    candidates = []
    for person in by_person.values():
        person["similarity"] = max(person["scores"])
        candidates.append(person)

    candidates.sort(
        key=lambda person: person["similarity"],
        reverse=True
    )

    if not candidates:
        return None, None

    best = candidates[0]
    runner_up_score = (
        candidates[1]["similarity"]
        if len(candidates) > 1
        else 0.0
    )
    margin = best["similarity"] - runner_up_score

    if (
        best["similarity"] < IDENTITY_THRESHOLD
        or (len(candidates) > 1 and margin < MINIMUM_MATCH_MARGIN)
    ):
        return None, {
            "similarity": best["similarity"],
            "margin": margin,
            "candidates": candidates
        }

    return best, {
        "similarity": best["similarity"],
        "margin": margin,
        "candidates": candidates
    }


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

            owner_username TEXT,

            latitude REAL,

            longitude REAL,

            accuracy REAL,

            status TEXT NOT NULL DEFAULT 'closed',

            tracking_active INTEGER NOT NULL DEFAULT 0,

            location_updated_at REAL,

            closed_at TIMESTAMP,

            updated_at TIMESTAMP,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    emergency_columns = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(emergencies)"
        ).fetchall()
    }

    emergency_migrations = {
        "owner_username": "TEXT",
        "latitude": "REAL",
        "longitude": "REAL",
        "accuracy": "REAL",
        "status": "TEXT NOT NULL DEFAULT 'closed'",
        "tracking_active": "INTEGER NOT NULL DEFAULT 0",
        "location_updated_at": "REAL",
        "closed_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
    }

    for column_name, column_type in emergency_migrations.items():
        if column_name not in emergency_columns:
            conn.execute(
                f"ALTER TABLE emergencies ADD COLUMN {column_name} {column_type}"
            )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS emergency_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            accuracy REAL NOT NULL,
            recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (incident_id) REFERENCES emergencies(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_emergency_locations_incident
        ON emergency_locations(incident_id, recorded_at)
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

            embedding TEXT,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (person_id)
            REFERENCES persons(id)
        )
    """)

    conn.commit()

    # --------------------------------------------------------
    # RESPONDERS TABLE
    # --------------------------------------------------------

    responders_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(responders)").fetchall()
    }

    conn.execute("""
        CREATE TABLE IF NOT EXISTS responders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            responder_type TEXT NOT NULL,
            phone TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            availability_status TEXT NOT NULL DEFAULT 'Available',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    responder_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(responders)").fetchall()
    }

    responder_migrations = {
        "name": "TEXT NOT NULL",
        "responder_type": "TEXT NOT NULL",
        "phone": "TEXT NOT NULL",
        "latitude": "REAL NOT NULL",
        "longitude": "REAL NOT NULL",
        "availability_status": "TEXT NOT NULL DEFAULT 'Available'",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "location_updated_at": "REAL"
    }

    for column_name, column_type in responder_migrations.items():
        if column_name not in responder_columns:
            conn.execute(
                f"ALTER TABLE responders ADD COLUMN {column_name} {column_type}"
            )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS responder_accounts (
            responder_id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (responder_id) REFERENCES responders(id)
        )
    """)

    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS emergency_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emergency_id INTEGER NOT NULL,
            responder_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN (
                    'Assigned',
                    'Accepted',
                    'Declined',
                    'On the way',
                    'Arrived',
                    'Resolved',
                    'Cancelled'
                )
            ),
            assigned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            accepted_at TIMESTAMP,
            declined_at TIMESTAMP,
            on_the_way_at TIMESTAMP,
            arrived_at TIMESTAMP,
            resolved_at TIMESTAMP,
            cancelled_at TIMESTAMP,
            FOREIGN KEY (emergency_id) REFERENCES emergencies(id) ON DELETE RESTRICT,
            FOREIGN KEY (responder_id) REFERENCES responders(id) ON DELETE RESTRICT
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_assignment_per_emergency
        ON emergency_assignments(emergency_id)
        WHERE status IN ('Assigned', 'Accepted', 'On the way', 'Arrived')
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_assignment_per_responder
        ON emergency_assignments(responder_id)
        WHERE status IN ('Assigned', 'Accepted', 'On the way', 'Arrived')
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_assignments_responder_history
        ON emergency_assignments(responder_id, assigned_at DESC)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assignment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            previous_status TEXT CHECK (
                previous_status IS NULL OR previous_status IN (
                    'Assigned',
                    'Accepted',
                    'Declined',
                    'On the way',
                    'Arrived',
                    'Resolved',
                    'Cancelled'
                )
            ),
            new_status TEXT NOT NULL CHECK (
                new_status IN (
                    'Assigned',
                    'Accepted',
                    'Declined',
                    'On the way',
                    'Arrived',
                    'Resolved',
                    'Cancelled'
                )
            ),
            actor_role TEXT NOT NULL CHECK (
                actor_role IN ('system', 'user', 'responder')
            ),
            actor_responder_id INTEGER,
            actor_username TEXT,
            changed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assignment_id)
                REFERENCES emergency_assignments(id) ON DELETE RESTRICT,
            FOREIGN KEY (actor_responder_id)
                REFERENCES responders(id) ON DELETE RESTRICT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_assignment_history_assignment
        ON assignment_history(assignment_id, changed_at)
    """)
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, name)
        VALUES (1, 'responder_location_freshness')
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, name)
        VALUES (2, 'persistent_emergency_assignments')
        """
    )

    conn.commit()

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
    ensure_face_embedding_schema()

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

            session.clear()
            session["username"] = user["username"]
            session["role"] = "user"

            return redirect(
                url_for("dashboard")
            )


        flash(
            "Invalid username or password.",
            "error"
        )

    elif session.get("role") == "responder":
        return redirect(url_for("responder_dashboard"))


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

    identification_count = conn.execute(
        "SELECT COUNT(*) FROM identifications"
    ).fetchone()[0]

    active_emergencies = get_active_emergencies(conn)
    active_emergency_count = len(active_emergencies)
    active_tracking_count = sum(
        int(emergency["tracking_active"])
        for emergency in active_emergencies
    )

    activity_events = conn.execute(
        """
        SELECT *
        FROM (
            SELECT
                'emergency' AS event_type,
                id AS event_id,
                emergency_type AS title,
                person_name AS subject,
                location AS location,
                status AS status,
                created_at AS event_at
            FROM emergencies
            UNION ALL
            SELECT
                'identification' AS event_type,
                id AS event_id,
                'Person identified' AS title,
                person_name AS subject,
                NULL AS location,
                'verified' AS status,
                identified_at AS event_at
            FROM identifications
        )
        ORDER BY event_at DESC
        LIMIT 6
        """
    ).fetchall()


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

        identification_count=identification_count,

        activity_events=activity_events,

        last_identified_person=last_identified_person,

        active_emergencies=active_emergencies,

        active_emergency_count=active_emergency_count,

        active_tracking_count=active_tracking_count
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

    conn = get_db()
    registered_persons = conn.execute(
        "SELECT COUNT(*) FROM persons"
    ).fetchone()[0]
    recent_identifications = conn.execute(
        """
        SELECT person_name, confidence, identified_at
        FROM identifications
        ORDER BY identified_at DESC
        LIMIT 10
        """
    ).fetchall()
    conn.close()

    return render_template(
        "identify.html",
        username=session["username"],
        registered_persons=registered_persons,
        recent_identifications=recent_identifications
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


    conn = get_db()
    active_emergencies = get_active_emergencies(
        conn,
        session["username"]
    )
    active_emergency = (
        dict(active_emergencies[0])
        if active_emergencies
        else None
    )
    conn.close()

    return render_template(
        "emergency.html",
        username=session["username"],
        active_emergency=active_emergency
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


    conn = get_db()
    active_emergencies = [
        dict(row)
        for row in get_active_emergencies(conn)
    ]
    conn.close()

    return render_template(
        "responders.html",
        username=session["username"],
        active_emergencies=active_emergencies
    )


@app.route("/responder/login", methods=["GET", "POST"])
def responder_login():

    if session.get("role") == "responder":
        return redirect(url_for("responder_dashboard"))
    if session.get("role") == "user":
        session.clear()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        try:
            account = conn.execute(
                """
                SELECT
                    responder_accounts.responder_id,
                    responder_accounts.username,
                    responder_accounts.password_hash
                FROM responder_accounts
                JOIN responders
                  ON responders.id = responder_accounts.responder_id
                WHERE responder_accounts.username = ? COLLATE NOCASE
                """,
                (username,)
            ).fetchone()
        finally:
            conn.close()

        if account and check_password_hash(account["password_hash"], password):
            session.clear()
            session["role"] = "responder"
            session["responder_id"] = account["responder_id"]
            session["responder_username"] = account["username"]
            return redirect(url_for("responder_dashboard"))

        flash("Invalid responder username or password.", "error")

    return render_template("responder_login.html")


@app.route("/responder/register", methods=["GET", "POST"])
def responder_register():

    if session.get("role") == "responder":
        return redirect(url_for("responder_dashboard"))
    if session.get("role") == "user":
        session.clear()

    if "responder_register_csrf" not in session:
        session["responder_register_csrf"] = secrets.token_urlsafe(32)

    if request.method == "POST":
        submitted_token = request.form.get("csrf_token", "")
        expected_token = session.get("responder_register_csrf", "")
        if not submitted_token or not secrets.compare_digest(submitted_token, expected_token):
            flash("Registration form expired. Please try again.", "error")
            return redirect(url_for("responder_register"))

        name = request.form.get("name", "").strip()
        responder_type = request.form.get("responder_type", "").strip()
        phone = request.form.get("phone", "").strip()
        username = name
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or len(name) > 120:
            flash("Enter a name with no more than 120 characters.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            )
        allowed_types = {
            "Police",
            "Ambulance",
            "Fire & Rescue",
            "Medical",
            "Disaster Response"
        }
        if responder_type not in allowed_types:
            flash("Choose a valid responder service type.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            )
        if not phone or len(phone) > 40:
            flash("Enter a phone number with no more than 40 characters.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            )
        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            )
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            )

        coords, coord_error = validate_latitude_longitude(
            request.form.get("latitude"),
            request.form.get("longitude")
        )
        if coord_error:
            flash("Acquire your device location before registering.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            )

        conn = get_db()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing_name = conn.execute(
                """
                SELECT 1 FROM responders WHERE name = ? COLLATE NOCASE
                UNION ALL
                SELECT 1 FROM responder_accounts WHERE username = ? COLLATE NOCASE
                UNION ALL
                SELECT 1 FROM users WHERE username = ? COLLATE NOCASE
                LIMIT 1
                """,
                (name, username, username)
            ).fetchone()
            if existing_name is not None:
                conn.rollback()
                flash("That responder name is already in use. Please contact GERIS support if you are already registered.", "error")
                return render_template(
                    "responder_register.html",
                    csrf_token=session["responder_register_csrf"]
                )

            cursor = conn.execute(
                """
                INSERT INTO responders (
                    name,
                    responder_type,
                    phone,
                    latitude,
                    longitude,
                    availability_status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'Offline', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (name, responder_type, phone, coords[0], coords[1])
            )
            responder_id = cursor.lastrowid
            conn.execute(
                """
                INSERT INTO responder_accounts (responder_id, username, password_hash)
                VALUES (?, ?, ?)
                """,
                (responder_id, username, generate_password_hash(password))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
            app.logger.exception("Unable to register responder account")
            flash("Registration could not be completed. Check the responder details and try again.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            ), 409
        except sqlite3.Error:
            conn.rollback()
            app.logger.exception("Unable to register responder account")
            flash("Unable to save responder registration. Please try again.", "error")
            return render_template(
                "responder_register.html",
                csrf_token=session["responder_register_csrf"]
            ), 500
        finally:
            conn.close()

        session.pop("responder_register_csrf", None)
        flash("Responder account created. Sign in to view your dashboard.", "success")
        return redirect(url_for("responder_login"))

    return render_template(
        "responder_register.html",
        csrf_token=session["responder_register_csrf"]
    )


@app.route("/responder/dashboard")
def responder_dashboard():

    responder_id = session.get("responder_id")
    responder_username = session.get("responder_username")
    if (
        session.get("role") != "responder"
        or not responder_id
        or not responder_username
    ):
        return redirect(url_for("responder_login"))

    conn = get_db()
    try:
        responder = conn.execute(
            """
            SELECT
                responders.id,
                responders.name,
                responders.responder_type,
                responders.phone,
                responders.availability_status,
                responders.location_updated_at
            FROM responders
            JOIN responder_accounts
              ON responder_accounts.responder_id = responders.id
            WHERE responders.id = ?
              AND responder_accounts.username = ? COLLATE NOCASE
            """,
            (responder_id, responder_username)
        ).fetchone()
        assignments = []
        if responder is not None:
            assignments = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        assignments.id,
                        emergencies.id AS emergency_id,
                        assignments.status,
                        assignments.assigned_at,
                        assignments.accepted_at,
                        assignments.declined_at,
                        assignments.on_the_way_at,
                        assignments.arrived_at,
                        assignments.resolved_at,
                        assignments.cancelled_at,
                        emergencies.emergency_type,
                        emergencies.created_at AS emergency_created_at,
                        emergencies.status AS emergency_status,
                        emergencies.latitude,
                        emergencies.longitude
                    FROM emergency_assignments AS assignments
                    JOIN emergencies
                      ON emergencies.id = assignments.emergency_id
                    WHERE assignments.responder_id = ?
                    ORDER BY assignments.assigned_at DESC, assignments.id DESC
                    """,
                    (responder_id,)
                ).fetchall()
            ]
    finally:
        conn.close()

    if responder is None:
        session.clear()
        flash("Responder profile not found. Please contact the GERIS administrator.", "error")
        return redirect(url_for("responder_login"))

    location_is_fresh = (
        responder["location_updated_at"] is not None
        and 0 <= time.time() - responder["location_updated_at"]
        <= RESPONDER_LOCATION_MAX_AGE_SECONDS
    )
    for assignment in assignments:
        coords, coord_error = validate_latitude_longitude(
            assignment["latitude"],
            assignment["longitude"]
        )
        assignment["map_url"] = None
        if not coord_error:
            latitude, longitude = coords
            assignment["map_url"] = (
                "https://www.openstreetmap.org/"
                f"?mlat={latitude}&mlon={longitude}"
                f"#map=16/{latitude}/{longitude}"
            )
        assignment["next_statuses"] = sorted(
            ASSIGNMENT_STATUS_TRANSITIONS.get(assignment["status"], ())
        )

    return render_template(
        "responder_dashboard.html",
        responder=dict(responder),
        username=responder_username,
        assignments=assignments,
        location_is_fresh=location_is_fresh
    )


@app.route("/api/responder/location", methods=["POST"])
def update_own_responder_location():
    if session.get("role") != "responder" or not session.get("responder_id"):
        return jsonify({
            "success": False,
            "message": "Responder sign-in is required."
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No location update received."
        }), 400

    coords, coord_error = validate_location(data)
    if coord_error:
        return jsonify({
            "success": False,
            "message": coord_error
        }), 400

    now = time.time()
    conn = get_db()
    try:
        cursor = conn.execute(
            """
            UPDATE responders
            SET latitude = ?,
                longitude = ?,
                location_updated_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND EXISTS (
                  SELECT 1
                  FROM responder_accounts
                  WHERE responder_accounts.responder_id = responders.id
                    AND responder_accounts.username = ? COLLATE NOCASE
              )
            """,
            (
                coords[0],
                coords[1],
                now,
                session["responder_id"],
                session["responder_username"]
            )
        )
        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "message": "Responder profile not found."
            }), 404
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to update authenticated responder location")
        return jsonify({
            "success": False,
            "message": "Unable to save your current location."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "location_updated_at": now,
        "message": "Your current location was updated."
    })


@app.route(
    "/api/responder/assignments/<int:assignment_id>/status",
    methods=["PUT"]
)
def update_own_assignment_status(assignment_id):
    if session.get("role") != "responder" or not session.get("responder_id"):
        return jsonify({
            "success": False,
            "message": "Responder sign-in is required."
        }), 401

    data = request.get_json(silent=True)
    new_status = data.get("status") if isinstance(data, dict) else None
    if not isinstance(new_status, str) or not new_status.strip():
        return jsonify({
            "success": False,
            "message": "A new assignment status is required."
        }), 400
    new_status = new_status.strip()

    timestamp_columns = {
        "Accepted": "accepted_at",
        "Declined": "declined_at",
        "On the way": "on_the_way_at",
        "Arrived": "arrived_at",
        "Resolved": "resolved_at"
    }
    if new_status not in timestamp_columns:
        return jsonify({
            "success": False,
            "message": "Assignment status is not supported."
        }), 400

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assignment = conn.execute(
            """
            SELECT
                emergency_assignments.id,
                emergency_assignments.responder_id,
                emergency_assignments.emergency_id,
                emergency_assignments.status
            FROM emergency_assignments
            JOIN responder_accounts
              ON responder_accounts.responder_id =
                 emergency_assignments.responder_id
            WHERE emergency_assignments.id = ?
              AND emergency_assignments.responder_id = ?
              AND responder_accounts.username = ? COLLATE NOCASE
            """,
            (
                assignment_id,
                session["responder_id"],
                session["responder_username"]
            )
        ).fetchone()
        if assignment is None:
            conn.rollback()
            return jsonify({
                "success": False,
                "message": "Assignment not found."
            }), 404

        allowed_next_statuses = ASSIGNMENT_STATUS_TRANSITIONS.get(
            assignment["status"],
            set()
        )
        if new_status not in allowed_next_statuses:
            conn.rollback()
            return jsonify({
                "success": False,
                "message": (
                    f"Cannot change assignment from {assignment['status']} "
                    f"to {new_status}."
                )
            }), 409

        emergency = conn.execute(
            "SELECT status FROM emergencies WHERE id = ?",
            (assignment["emergency_id"],)
        ).fetchone()
        if emergency is None or emergency["status"] != "active":
            conn.rollback()
            return jsonify({
                "success": False,
                "message": "This emergency is no longer active."
            }), 409

        timestamp_column = timestamp_columns[new_status]
        conn.execute(
            f"""
            UPDATE emergency_assignments
            SET status = ?,
                {timestamp_column} = CURRENT_TIMESTAMP
            WHERE id = ? AND responder_id = ?
            """,
            (new_status, assignment_id, session["responder_id"])
        )
        if new_status in {"Declined", "Resolved"}:
            conn.execute(
                """
                UPDATE responders
                SET availability_status = 'Available',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND availability_status = 'Busy'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM emergency_assignments
                      WHERE responder_id = ?
                        AND status IN (
                            'Assigned', 'Accepted', 'On the way', 'Arrived'
                        )
                  )
                """,
                (session["responder_id"], session["responder_id"])
            )
        add_assignment_history(
            conn,
            assignment_id,
            assignment["status"],
            new_status,
            "responder",
            actor_responder_id=session["responder_id"]
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to update responder assignment status")
        return jsonify({
            "success": False,
            "message": "Unable to update assignment status."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "status": new_status,
        "message": f"Assignment status updated to {new_status}."
    })


@app.route("/responder/logout")
def responder_logout():

    session.clear()
    return redirect(url_for("responder_login"))


# ============================================================
# RESPONDER API FOUNDATION
# ============================================================

@app.route("/api/responders", methods=["POST"])
def create_responder():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No responder data received."
        }), 400

    required_fields = [
        "name",
        "responder_type",
        "phone",
        "latitude",
        "longitude",
        "availability_status"
    ]
    missing = [field for field in required_fields if field not in data or data.get(field) in (None, "")]
    if missing:
        return jsonify({
            "success": False,
            "message": "Missing required responder fields: " + ", ".join(missing)
        }), 400

    name = str(data.get("name", "")).strip()
    responder_type = str(data.get("responder_type", "")).strip()
    phone = str(data.get("phone", "")).strip()
    availability_status, status_error = normalize_responder_status(data.get("availability_status"))
    if not name:
        return jsonify({
            "success": False,
            "message": "Name is required."
        }), 400

    if not responder_type:
        return jsonify({
            "success": False,
            "message": "Responder type is required."
        }), 400

    if not phone:
        return jsonify({
            "success": False,
            "message": "Phone is required."
        }), 400

    if status_error:
        return jsonify({
            "success": False,
            "message": status_error
        }), 400

    coords, coord_error = validate_latitude_longitude(
        data.get("latitude"),
        data.get("longitude")
    )
    if coord_error:
        return jsonify({
            "success": False,
            "message": coord_error
        }), 400

    latitude, longitude = coords
    conn = get_db()
    try:
        cursor = conn.execute(
            """
            INSERT INTO responders (
                name,
                responder_type,
                phone,
                latitude,
                longitude,
                availability_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                name,
                responder_type,
                phone,
                latitude,
                longitude,
                availability_status
            )
        )
        responder_id = cursor.lastrowid
        responder = conn.execute(
            "SELECT * FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to create responder")
        return jsonify({
            "success": False,
            "message": "Unable to save responder."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Responder created successfully.",
        "responder": dict(responder)
    }), 201


@app.route("/api/responders/<int:responder_id>/account", methods=["POST"])
def create_responder_account(responder_id):

    if session.get("role") != "user" or "username" not in session:
        return jsonify({
            "success": False,
            "message": "A GERIS user session is required to provision responder access."
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "Responder account credentials are required."
        }), 400

    username = data.get("username")
    password = data.get("password")
    if not isinstance(username, str) or len(username.strip()) < 3:
        return jsonify({
            "success": False,
            "message": "Responder username must contain at least 3 characters."
        }), 400
    if not isinstance(password, str) or len(password) < 8:
        return jsonify({
            "success": False,
            "message": "Responder password must contain at least 8 characters."
        }), 400

    username = username.strip()
    conn = get_db()
    try:
        responder = conn.execute(
            "SELECT id FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()
        if responder is None:
            return jsonify({
                "success": False,
                "message": "Responder not found."
            }), 404

        existing_account = conn.execute(
            "SELECT responder_id FROM responder_accounts WHERE responder_id = ?",
            (responder_id,)
        ).fetchone()
        if existing_account is not None:
            return jsonify({
                "success": False,
                "message": "This responder already has a login account."
            }), 409

        existing_username = conn.execute(
            """
            SELECT 1 FROM responder_accounts WHERE username = ? COLLATE NOCASE
            UNION ALL
            SELECT 1 FROM users WHERE username = ? COLLATE NOCASE
            LIMIT 1
            """,
            (username, username)
        ).fetchone()
        if existing_username is not None:
            return jsonify({
                "success": False,
                "message": "That username is already in use."
            }), 409

        conn.execute(
            """
            INSERT INTO responder_accounts (responder_id, username, password_hash)
            VALUES (?, ?, ?)
            """,
            (responder_id, username, generate_password_hash(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        app.logger.exception("Unable to provision responder account")
        return jsonify({
            "success": False,
            "message": "Responder account could not be created because the username or responder account is already in use."
        }), 409
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to provision responder account")
        return jsonify({
            "success": False,
            "message": "Unable to create responder account."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Responder login created successfully.",
        "responder_id": responder_id,
        "username": username
    }), 201


@app.route("/api/responders", methods=["GET"])
def get_responders():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    conn = get_db()
    try:
        responders = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM responders
                ORDER BY created_at DESC
                """
            ).fetchall()
        ]
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "responders": responders
    })


@app.route("/api/responders/<int:responder_id>", methods=["GET"])
def get_responder_by_id(responder_id):

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    conn = get_db()
    try:
        responder = conn.execute(
            "SELECT * FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()
    finally:
        conn.close()

    if responder is None:
        return jsonify({
            "success": False,
            "message": "Responder not found."
        }), 404

    return jsonify({
        "success": True,
        "responder": dict(responder)
    })


@app.route("/api/responders/<int:responder_id>", methods=["PUT"])
def update_responder(responder_id):

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No responder data received."
        }), 400

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()

        if existing is None:
            return jsonify({
                "success": False,
                "message": "Responder not found."
            }), 404

        fields = []
        values = []

        if "name" in data:
            name = str(data.get("name", "")).strip()
            if not name:
                return jsonify({
                    "success": False,
                    "message": "Name is required."
                }), 400
            fields.append("name = ?")
            values.append(name)

        if "responder_type" in data:
            responder_type = str(data.get("responder_type", "")).strip()
            if not responder_type:
                return jsonify({
                    "success": False,
                    "message": "Responder type is required."
                }), 400
            fields.append("responder_type = ?")
            values.append(responder_type)

        if "phone" in data:
            phone = str(data.get("phone", "")).strip()
            if not phone:
                return jsonify({
                    "success": False,
                    "message": "Phone is required."
                }), 400
            fields.append("phone = ?")
            values.append(phone)

        if "availability_status" in data:
            availability_status, status_error = normalize_responder_status(data.get("availability_status"))
            if status_error:
                return jsonify({
                    "success": False,
                    "message": status_error
                }), 400
            has_active_assignment = conn.execute(
                """
                SELECT 1
                FROM emergency_assignments
                WHERE responder_id = ?
                  AND status IN (
                      'Assigned', 'Accepted', 'On the way', 'Arrived'
                  )
                LIMIT 1
                """,
                (responder_id,)
            ).fetchone()
            if has_active_assignment and availability_status != "Busy":
                return jsonify({
                    "success": False,
                    "message": "Responder availability is controlled by the active assignment."
                }), 409
            fields.append("availability_status = ?")
            values.append(availability_status)

        if "latitude" in data or "longitude" in data:
            latitude = data.get("latitude", existing["latitude"])
            longitude = data.get("longitude", existing["longitude"])
            coords, coord_error = validate_latitude_longitude(latitude, longitude)
            if coord_error:
                return jsonify({
                    "success": False,
                    "message": coord_error
                }), 400
            fields.append("latitude = ?")
            values.append(coords[0])
            fields.append("longitude = ?")
            values.append(coords[1])

        if not fields:
            return jsonify({
                "success": False,
                "message": "No responder fields were provided to update."
            }), 400

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(responder_id)

        cursor = conn.execute(
            f"UPDATE responders SET {', '.join(fields)} WHERE id = ?",
            values
        )
        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "message": "Responder not found."
            }), 404

        responder = conn.execute(
            "SELECT * FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to update responder")
        return jsonify({
            "success": False,
            "message": "Unable to update responder."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Responder updated successfully.",
        "responder": dict(responder)
    })


@app.route("/api/responders/<int:responder_id>/location", methods=["PUT"])
def update_responder_location(responder_id):

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No responder location data received."
        }), 400

    latitude = data.get("latitude")
    longitude = data.get("longitude")
    if latitude is None or longitude is None:
        return jsonify({
            "success": False,
            "message": "Latitude and longitude are required."
        }), 400

    coords, coord_error = validate_latitude_longitude(latitude, longitude)
    if coord_error:
        return jsonify({
            "success": False,
            "message": coord_error
        }), 400

    conn = get_db()
    try:
        cursor = conn.execute(
            """
            UPDATE responders
            SET latitude = ?,
                longitude = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (coords[0], coords[1], responder_id)
        )
        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "message": "Responder not found."
            }), 404

        responder = conn.execute(
            "SELECT * FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to update responder location")
        return jsonify({
            "success": False,
            "message": "Unable to update responder location."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Responder location updated successfully.",
        "responder": dict(responder)
    })


@app.route("/api/responders/<int:responder_id>/status", methods=["PUT"])
def update_responder_status(responder_id):

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No responder status data received."
        }), 400

    availability_status, status_error = normalize_responder_status(data.get("availability_status"))
    if status_error:
        return jsonify({
            "success": False,
            "message": status_error
        }), 400

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        has_active_assignment = conn.execute(
            """
            SELECT 1
            FROM emergency_assignments
            WHERE responder_id = ?
              AND status IN (
                  'Assigned', 'Accepted', 'On the way', 'Arrived'
              )
            LIMIT 1
            """,
            (responder_id,)
        ).fetchone()
        if has_active_assignment and availability_status != "Busy":
            conn.rollback()
            return jsonify({
                "success": False,
                "message": "Responder availability is controlled by the active assignment."
            }), 409

        cursor = conn.execute(
            """
            UPDATE responders
            SET availability_status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (availability_status, responder_id)
        )
        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "message": "Responder not found."
            }), 404

        responder = conn.execute(
            "SELECT * FROM responders WHERE id = ?",
            (responder_id,)
        ).fetchone()
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to update responder status")
        return jsonify({
            "success": False,
            "message": "Unable to update responder status."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Responder status updated successfully.",
        "responder": dict(responder)
    })


@app.route("/api/responders/nearby", methods=["GET", "POST"])
def nearest_responders():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    payload = request.get_json(silent=True) if request.method == "POST" else {}
    if not isinstance(payload, dict):
        payload = {}

    latitude_value = request.args.get("latitude", payload.get("latitude"))
    longitude_value = request.args.get("longitude", payload.get("longitude"))
    limit_value = request.args.get("limit", payload.get("limit", 5))

    try:
        limit = int(limit_value)
    except (TypeError, ValueError):
        limit = 5

    limit = max(1, min(limit, 20))

    coords, coord_error = validate_latitude_longitude(latitude_value, longitude_value)
    if coord_error:
        return jsonify({
            "success": False,
            "message": coord_error
        }), 400

    latitude, longitude = coords

    conn = get_db()
    try:
        responder_rows = conn.execute(
            """
            SELECT *
            FROM responders
            WHERE availability_status = 'Available'
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    candidates = []
    for row in responder_rows:
        responder_latitude = float(row["latitude"])
        responder_longitude = float(row["longitude"])
        distance = distance_meters(
            latitude,
            longitude,
            responder_latitude,
            responder_longitude
        )
        candidates.append({
            "id": row["id"],
            "name": row["name"],
            "responder_type": row["responder_type"],
            "phone": row["phone"],
            "latitude": responder_latitude,
            "longitude": responder_longitude,
            "availability_status": row["availability_status"],
            "distance_meters": round(distance, 2),
            "distance_km": round(distance / 1000, 3)
        })

    candidates.sort(key=lambda item: (item["distance_meters"], item["id"]))
    nearest = candidates[:limit]

    return jsonify({
        "success": True,
        "source": {
            "latitude": latitude,
            "longitude": longitude
        },
        "count": len(nearest),
        "responders": nearest
    })


@app.route("/api/emergency/nearest-responders", methods=["GET", "POST"])
def emergency_nearest_responders():
    return nearest_responders()


@app.route("/api/emergency/start", methods=["POST"])
def start_emergency_tracking():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No emergency data received."
        }), 400

    location_values, location_error = validate_location(data)
    if location_error:
        return jsonify({
            "success": False,
            "message": location_error
        }), 400

    person_name = data.get("person_name", "")
    contact_number = data.get("contact_number", "")
    emergency_type = data.get("emergency_type", "")
    description = data.get("description", "")
    address = data.get("location", "")

    if not all(
        isinstance(value, str)
        for value in (
            person_name,
            contact_number,
            emergency_type,
            description,
            address
        )
    ):
        return jsonify({
            "success": False,
            "message": "Emergency details must be text."
        }), 400

    person_name = person_name.strip()
    contact_number = contact_number.strip()
    emergency_type = emergency_type.strip()
    description = description.strip()
    address = address.strip()

    if not person_name or not emergency_type:
        return jsonify({
            "success": False,
            "message": "Person name and emergency type are required."
        }), 400

    latitude, longitude, accuracy = location_values
    coordinates_text = f"{latitude:.6f}, {longitude:.6f}"
    location_text = (
        f"{address} ({coordinates_text})"
        if address
        else coordinates_text
    )

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            INSERT INTO emergencies (
                person_name,
                contact_number,
                emergency_type,
                location,
                description,
                owner_username,
                latitude,
                longitude,
                accuracy,
                status,
                tracking_active,
                location_updated_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, CURRENT_TIMESTAMP)
            """,
            (
                person_name,
                contact_number,
                emergency_type,
                location_text,
                description,
                session["username"],
                latitude,
                longitude,
                accuracy,
                time.time()
            )
        )
        incident_id = cursor.lastrowid
        conn.execute(
            """
            INSERT INTO emergency_locations (
                incident_id,
                latitude,
                longitude,
                accuracy
            )
            VALUES (?, ?, ?, ?)
            """,
            (incident_id, latitude, longitude, accuracy)
        )
        assignment_result = try_assign_emergency(
            conn,
            incident_id,
            time.time()
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to save emergency tracking incident")
        return jsonify({
            "success": False,
            "message": "Unable to save the emergency. Please try again."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "incident_id": incident_id,
        "assignment": assignment_result,
        "message": "Emergency tracking started."
    })


@app.route(
    "/api/emergency/<int:incident_id>/dispatch",
    methods=["POST"]
)
def dispatch_existing_emergency(incident_id):
    if session.get("role") != "user" or "username" not in session:
        return jsonify({
            "success": False,
            "message": "A GERIS user session is required."
        }), 401

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        incident = conn.execute(
            """
            SELECT id
            FROM emergencies
            WHERE id = ?
              AND owner_username = ?
              AND status = 'active'
            """,
            (incident_id, session["username"])
        ).fetchone()
        if incident is None:
            conn.rollback()
            return jsonify({
                "success": False,
                "message": "Active emergency not found."
            }), 404

        assignment_result = try_assign_emergency(
            conn,
            incident_id,
            time.time()
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to dispatch emergency")
        return jsonify({
            "success": False,
            "message": "Unable to attempt responder assignment."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "assignment": assignment_result
    })


@app.route("/api/emergency/location", methods=["POST"])
def update_emergency_location():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "No location update received."
        }), 400

    try:
        incident_id = int(data.get("incident_id"))
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Invalid emergency incident."
        }), 400

    if incident_id <= 0:
        return jsonify({
            "success": False,
            "message": "Invalid emergency incident."
        }), 400

    location_values, location_error = validate_location(data)
    if location_error:
        return jsonify({
            "success": False,
            "message": location_error
        }), 400

    latitude, longitude, accuracy = location_values
    now = time.time()
    conn = get_db()

    try:
        incident = conn.execute(
            """
            SELECT latitude, longitude, location_updated_at
            FROM emergencies
            WHERE id = ?
              AND owner_username = ?
              AND status = 'active'
              AND tracking_active = 1
            """,
            (incident_id, session["username"])
        ).fetchone()

        if incident is None:
            return jsonify({
                "success": False,
                "message": "No active emergency tracking session was found."
            }), 404

        movement = distance_meters(
            incident["latitude"],
            incident["longitude"],
            latitude,
            longitude
        )
        last_update = incident["location_updated_at"] or 0
        if movement < 25 and now - last_update < 15:
            return jsonify({
                "success": True,
                "updated": False,
                "message": "Location unchanged within the update interval."
            })

        coordinates_text = f"{latitude:.6f}, {longitude:.6f}"
        conn.execute(
            """
            UPDATE emergencies
            SET latitude = ?,
                longitude = ?,
                accuracy = ?,
                location = ?,
                location_updated_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                latitude,
                longitude,
                accuracy,
                coordinates_text,
                now,
                incident_id
            )
        )

        last_history = conn.execute(
            """
            SELECT latitude, longitude, recorded_at
            FROM emergency_locations
            WHERE incident_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (incident_id,)
        ).fetchone()
        add_history = last_history is None
        if last_history is not None:
            history_movement = distance_meters(
                last_history["latitude"],
                last_history["longitude"],
                latitude,
                longitude
            )
            elapsed = conn.execute(
                """
                SELECT
                    (julianday('now') - julianday(?)) * 86400
                """,
                (last_history["recorded_at"],)
            ).fetchone()[0] or 0
            add_history = history_movement >= 25 or elapsed >= 60

        if add_history:
            conn.execute(
                """
                INSERT INTO emergency_locations (
                    incident_id,
                    latitude,
                    longitude,
                    accuracy
                )
                VALUES (?, ?, ?, ?)
                """,
                (incident_id, latitude, longitude, accuracy)
            )

        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to update emergency location")
        return jsonify({
            "success": False,
            "message": "Unable to save the latest location."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "updated": True,
        "message": "Emergency location updated."
    })


@app.route("/api/emergency/stop", methods=["POST"])
def stop_emergency_tracking():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.get_json(silent=True)
    try:
        incident_id = int(data.get("incident_id")) if isinstance(data, dict) else 0
    except (TypeError, ValueError):
        incident_id = 0

    if incident_id <= 0:
        return jsonify({
            "success": False,
            "message": "Invalid emergency incident."
        }), 400

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE emergencies
            SET status = 'closed',
                tracking_active = 0,
                closed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND owner_username = ?
              AND status = 'active'
            """,
            (incident_id, session["username"])
        )
        if cursor.rowcount == 0:
            conn.rollback()
            return jsonify({
                "success": False,
                "message": "No active emergency tracking session was found."
            }), 404

        active_assignments = conn.execute(
            """
            SELECT id, responder_id, status
            FROM emergency_assignments
            WHERE emergency_id = ?
              AND status IN (
                  'Assigned', 'Accepted', 'On the way', 'Arrived'
              )
            """,
            (incident_id,)
        ).fetchall()
        for assignment in active_assignments:
            conn.execute(
                """
                UPDATE emergency_assignments
                SET status = 'Cancelled',
                    cancelled_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (assignment["id"],)
            )
            add_assignment_history(
                conn,
                assignment["id"],
                assignment["status"],
                "Cancelled",
                "user",
                actor_username=session["username"]
            )
            conn.execute(
                """
                UPDATE responders
                SET availability_status = 'Available',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND availability_status = 'Busy'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM emergency_assignments
                      WHERE responder_id = ?
                        AND status IN (
                            'Assigned', 'Accepted', 'On the way', 'Arrived'
                        )
                  )
                """,
                (assignment["responder_id"], assignment["responder_id"])
            )

        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        app.logger.exception("Unable to stop emergency tracking")
        return jsonify({
            "success": False,
            "message": "Unable to close the emergency."
        }), 500
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Emergency closed and location tracking stopped."
    })


@app.route("/api/emergency/active")
def active_emergency_locations():

    if "username" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    conn = get_db()
    incidents = [
        dict(row)
        for row in get_active_emergencies(conn)
    ]
    conn.close()

    return jsonify({
        "success": True,
        "incidents": incidents
    })


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

            image_data = face_image.split(
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
                flash(
                    "No face detected. Please position your face inside the camera frame.",
                    "error"
                )
                return render_template(
                    "register_person.html",
                    username=session["username"]
                )

            face_data = analyze_primary_face(image)

            if face_data is None:
                flash(
                    "No face detected. Please position your face inside the camera frame.",
                    "error"
                )
                return render_template(
                    "register_person.html",
                    username=session["username"]
                )

            person_dir = (
                BASE_DIR
                / "static"
                / "persons"
            )

            person_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            import uuid

            image_filename = (
                str(uuid.uuid4())
                + ".jpg"
            )

            image_path = (
                person_dir
                / image_filename
            )

            if not cv2.imwrite(str(image_path), face_data["crop"]):
                raise OSError("Could not save the detected face image.")

            conn = get_db()

            existing_person = conn.execute(
                """
                SELECT id
                FROM persons
                WHERE lower(name) = lower(?)
                LIMIT 1
                """,
                (name,)
            ).fetchone()

            is_existing_person = existing_person is not None

            if existing_person:

                person_id = existing_person["id"]

                conn.execute(
                    """
                    UPDATE persons
                    SET phone = ?,
                        emergency_contact = ?
                    WHERE id = ?
                    """,
                    (phone, emergency_contact, person_id)
                )

                conn.execute(
                    """
                    INSERT INTO face_samples
                    (person_id, image_path, embedding)
                    VALUES (?, ?, ?)
                    """,
                    (
                        person_id,
                        f"persons/{image_filename}",
                        serialize_embedding(face_data["embedding"])
                    )
                )

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

                conn.execute(
                    """
                    INSERT INTO face_samples
                    (person_id, image_path, embedding)
                    VALUES (?, ?, ?)
                    """,
                    (
                        person_id,
                        f"persons/{image_filename}",
                        serialize_embedding(face_data["embedding"])
                    )
                )

            conn.commit()
            conn.close()

            flash(
                "Additional face sample saved to the existing person record."
                if is_existing_person
                else "Person registered successfully.",
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
                "Unable to register person. Please try again.",
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

        image_data = data["image"]
        if "," not in image_data:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400
        image_data = image_data.split(",", 1)[1]

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


        face_data = analyze_primary_face(image)

        if face_data is None:
            return jsonify({
                "success": False,
                "face_detected": False,
                "message": "No face detected. Please position your face inside the camera frame."
            })

        registered_samples = load_registered_embeddings()
        if not registered_samples:
            return jsonify({
                "success": True,
                "identified": False,
                "face_detected": True,
                "message": "No registered person matched this face.",
                "status": "Not Identified"
            })

        matched_person, match_details = match_registered_person(
            face_data["embedding"],
            registered_samples
        )

        print()
        print("==============================")
        print("GERIS FACE RECOGNITION")
        print("==============================")
        print("Detected face:", "YES")
        print("Registered face samples:", len(registered_samples))
        print("Similarity threshold:", IDENTITY_THRESHOLD)
        print("Required match margin:", MINIMUM_MATCH_MARGIN)

        if match_details:
            print("Best similarity:", round(match_details["similarity"], 4))
            print("Best-vs-next margin:", round(match_details["margin"], 4))

        if matched_person is None:
            print("Result: NO MATCH")
            return jsonify({
                "success": True,
                "identified": False,
                "face_detected": True,
                "message": "No registered person matched this face.",
                "status": "Not Identified",
                "similarity": round(
                    match_details["similarity"],
                    4
                ) if match_details else None
            })

        similarity = matched_person["similarity"]
        conn = get_db()
        conn.execute(
            """
            INSERT INTO identifications
            (person_name, confidence)
            VALUES (?, ?)
            """,
            (
                matched_person["name"],
                similarity
            )
        )
        conn.commit()
        conn.close()

        print("Best match:", matched_person["name"])
        print("Result: MATCH")

        return jsonify({
            "success": True,
            "identified": True,
            "face_detected": True,
            "name": matched_person["name"],
            "person_id": matched_person["person_id"],
            "phone": matched_person["phone"],
            "emergency_contact": matched_person["emergency_contact"],
            "confidence": round(similarity * 100, 2),
            "similarity": round(similarity, 4),
            "status": "Verified",
            "message": "Person identified successfully."
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