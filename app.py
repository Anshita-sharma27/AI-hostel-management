from flask import Flask, render_template, request, redirect, jsonify
import sqlite3
import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime
from flask import Response
unknown_alert = False
app = Flask(__name__)

# ================= DATABASE =================

def connect_db():

    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row

    return conn


def create_tables():

    conn = connect_db()
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        room TEXT,
        department TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS complaints(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT,
        complaint TEXT,
        category TEXT,
        status TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT,
        time TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS mess_feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student TEXT,
        feedback TEXT,
        sentiment TEXT
    )
    ''')

    conn.commit()
    conn.close()


create_tables()


# ================= FACE RECOGNITION =================

known_face_encodings = []
known_face_names = []

face_folder = "known_faces"

if not os.path.exists(face_folder):
    os.makedirs(face_folder)

for file in os.listdir(face_folder):

    if file.endswith('.jpg') or file.endswith('.png'):

        image_path = os.path.join(
            face_folder,
            file
        )

        image = face_recognition.load_image_file(
            image_path
        )

        encodings = face_recognition.face_encodings(
            image
        )

        if len(encodings) > 0:

            known_face_encodings.append(
                encodings[0]
            )

            name = os.path.splitext(file)[0]

            known_face_names.append(
                name.lower()
            )

print("Known Faces Loaded:")
print(known_face_names)

# ================= NLP COMPLAINT ANALYSIS =================

def analyze_complaint(text):

    text = text.lower()

    if 'wifi' in text:
        return 'Internet Issue'

    elif 'water' in text:
        return 'Water Problem'

    elif 'electricity' in text:
        return 'Electricity Problem'

    elif 'food' in text:
        return 'Mess Issue'

    else:
        return 'General Complaint'

# ================= SENTIMENT ANALYSIS =================

def analyze_feedback(text):

    positive_words = [
        'good',
        'great',
        'excellent',
        'nice'
    ]

    negative_words = [
        'bad',
        'poor',
        'dirty',
        'worst'
    ]

    text = text.lower()

    for word in positive_words:

        if word in text:
            return 'Positive'

    for word in negative_words:

        if word in text:
            return 'Negative'

    return 'Neutral'

# ================= HOME =================

@app.route('/')
def index():

    return render_template('index.html')

# ================= REGISTER =================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        name = request.form['name'].strip().lower()
        room = request.form['room']
        department = request.form['department']

        conn = connect_db()

        existing = conn.execute(
            'SELECT * FROM students WHERE LOWER(name)=?',
            (name,)
        ).fetchone()

        if existing:
            return "Student already exists ⚠"

        conn.execute(
            'INSERT INTO students(name, room, department) VALUES (?, ?, ?)',
            (name, room, department)
        )

        conn.commit()
        conn.close()

        # 👉 FACE IMAGE SAVE PATH
        image = request.files['image']

        if image:

            path = f"known_faces/{name}.jpg"
            image.save(path)

        return "Registered Successfully ✅"

    return render_template('register.html')
# ================= ADMIN =================

@app.route('/admin')
def admin():

    conn = connect_db()

    students = conn.execute(
        'SELECT * FROM students'
    ).fetchall()

    complaints = conn.execute(
        'SELECT * FROM complaints'
    ).fetchall()

    attendance = conn.execute(
        'SELECT * FROM attendance'
    ).fetchall()

    feedbacks = conn.execute(
        'SELECT * FROM mess_feedback'
    ).fetchall()

    conn.close()

    return render_template(
        'admin.html',
        students=students,
        complaints=complaints,
        attendance=attendance,
        feedbacks=feedbacks
    )

# ================= COMPLAINT =================

@app.route('/complaint', methods=['GET', 'POST'])
def complaint():

    if request.method == 'POST':

        student = request.form['student']
        text = request.form['complaint']

        category = analyze_complaint(text)

        conn = connect_db()

        conn.execute(
            'INSERT INTO complaints(student, complaint, category, status) VALUES (?, ?, ?, ?)',
            (student, text, category, 'Pending')
        )

        conn.commit()
        conn.close()

        return redirect('/complaint')

    conn = connect_db()

    complaints = conn.execute(
        'SELECT * FROM complaints'
    ).fetchall()

    conn.close()

    return render_template(
        'complaints.html',
        complaints=complaints
    )

# ================= MESS =================

@app.route('/mess', methods=['GET', 'POST'])
def mess():

    if request.method == 'POST':

        student = request.form['student']
        feedback = request.form['feedback']

        sentiment = analyze_feedback(
            feedback
        )

        conn = connect_db()

        conn.execute(
            'INSERT INTO mess_feedback(student, feedback, sentiment) VALUES (?, ?, ?)',
            (student, feedback, sentiment)
        )

        conn.commit()
        conn.close()

        return redirect('/mess')

    conn = connect_db()

    feedbacks = conn.execute(
        'SELECT * FROM mess_feedback'
    ).fetchall()

    conn.close()

    return render_template(
        'mess.html',
        feedbacks=feedbacks
    )

# ================= ATTENDANCE PAGE =================

@app.route('/attendance')
def attendance():

    conn = connect_db()

    attendance = conn.execute(
        'SELECT * FROM attendance ORDER BY id DESC'
    ).fetchall()

    conn.close()

    return render_template(
        'attendance.html',
        attendance=attendance
    )

@app.route('/scan_face', methods=['POST'])
def scan_face():

    file = request.files.get('frame')

    if not file:

        return jsonify({
            'message': 'No Frame Received ❌'
        })

    npimg = np.frombuffer(file.read(), np.uint8)

    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(face_encodings) == 0:

        return jsonify({
            'message': 'No Face Detected ❌'
        })

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(face_distances) > 0:

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:

                student_name = known_face_names[best_match_index]

                conn = connect_db()

                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                conn.execute(
                    'INSERT INTO attendance(student, time) VALUES (?, ?)',
                    (student_name, current_time)
                )

                conn.commit()
                conn.close()

                return jsonify({
                    'message': 'Attendance Marked ✅',
                    'student': student_name
                })

    return jsonify({
        'message': 'Unknown Face ❌'
    })

    @app.route('/video_feed')
    def video_feed():

      camera = cv2.VideoCapture(0)

    def generate():

        while True:

            success, frame = camera.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

                cv2.putText(frame, name, (left + 10, bottom - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def load_faces():

    global known_face_encodings, known_face_names

    known_face_encodings = []
    known_face_names = []

    face_folder = "known_faces"

    if not os.path.exists(face_folder):
        os.makedirs(face_folder)

    for file in os.listdir(face_folder):

        if file.endswith('.jpg') or file.endswith('.png'):

            path = os.path.join(face_folder, file)

            image = face_recognition.load_image_file(path)

            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:

                known_face_encodings.append(encodings[0])

                name = os.path.splitext(file)[0].lower()

                known_face_names.append(name)
# ================= ANALYTICS =================

@app.route('/analytics')
def analytics():

    conn = connect_db()

    total_students = conn.execute(
        'SELECT COUNT(*) FROM students'
    ).fetchone()[0]

    total_complaints = conn.execute(
        'SELECT COUNT(*) FROM complaints'
    ).fetchone()[0]

    total_attendance = conn.execute(
        'SELECT COUNT(*) FROM attendance'
    ).fetchone()[0]

    conn.close()

    return jsonify({

        'students':
        total_students,

        'complaints':
        total_complaints,

        'attendance':
        total_attendance
    })

from openpyxl import Workbook
from flask import send_file
import os

@app.route('/download_attendance')
def download_attendance():

    conn = connect_db()

    data = conn.execute(
        'SELECT * FROM attendance ORDER BY id DESC'
    ).fetchall()

    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance Report"

    # HEADER
    ws.append(["ID", "Student Name", "Time"])

    # DATA
    for row in data:
        ws.append([row["id"], row["student"], row["time"]])

    file_path = "attendance_report.xlsx"
    wb.save(file_path)

    return send_file(file_path, as_attachment=True)
# ================= CCTV STREAM =================

def generate_frames():

    global unknown_alert

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()
        if not success:
            break

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            name = "Unknown"

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)

                if matches[best_match]:
                    name = known_face_names[best_match]

                    # Attendance save
                    conn = connect_db()
                    conn.execute(
                        "INSERT INTO attendance(student, time) VALUES (?, ?)",
                        (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    conn.commit()
                    conn.close()

                    unknown_alert = False

                else:
                    unknown_alert = True

            else:
                unknown_alert = True

            # BOX DRAW
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if name == "Unknown":
                cv2.putText(frame, "ALERT: UNKNOWN", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():

    camera = cv2.VideoCapture(0)

    def generate():

        while True:

            success, frame = camera.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(face_distances) > 0:

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

                cv2.putText(frame, name, (left + 10, bottom - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def alert_status():
    global unknown_alert
    return jsonify({"alert": unknown_alert})

@app.route('/add_student_face', methods=['POST'])
def add_student_face():

    name = request.form['name'].strip().lower()
    room = request.form['room']
    department = request.form['department']

    image = request.files['image']

    conn = connect_db()

    existing = conn.execute(
        'SELECT * FROM students WHERE LOWER(name)=?',
        (name,)
    ).fetchone()

    if existing:
        return "Student already exists ⚠"

    conn.execute(
        'INSERT INTO students(name, room, department) VALUES (?, ?, ?)',
        (name, room, department)
    )

    conn.commit()
    conn.close()

    if image:
        path = f"known_faces/{name}.jpg"
        image.save(path)

    load_faces()   # 🔥 reload faces instantly

    return "Student Added + Face Registered Successfully ✅"
 


def load_faces():

    global known_face_encodings, known_face_names

    known_face_encodings = []
    known_face_names = []

    face_folder = "known_faces"

    for file in os.listdir(face_folder):

        path = os.path.join(face_folder, file)

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:

            known_face_encodings.append(encodings[0])

            name = os.path.splitext(file)[0].lower()

            known_face_names.append(name)

# ================= MAIN =================

if __name__ == '__main__':

    load_faces()   # 🔥 MUST FIRST LOAD FACES
    app.run(debug=True)