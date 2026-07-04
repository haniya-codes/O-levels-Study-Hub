import os
import re
import json
import uuid
import base64
import mimetypes
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

try:
    import fitz  # PyMuPDF — used to turn PDF papers/submissions into images for the AI checker
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

load_dotenv()

# --------------------------------------------------------------------------- #
# Storage + config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).with_name("lecture_data")
VIDEO_DIR = DATA_DIR / "videos"
INDEX_FILE = DATA_DIR / "index.json"
DB_PATH = DATA_DIR / "school.db"          # <-- preference-system database lives alongside lecture data
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Paper checker storage — teacher-set papers + student submissions
PAPER_DIR = DATA_DIR / "papers"
SUBMISSION_DIR = DATA_DIR / "submissions"
PAPERS_INDEX_FILE = DATA_DIR / "papers_index.json"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
            "English", "Economics", "Accounting", "Islamiyat", "Pakistan Studies", "Urdu"]

VIDEO_TYPES = ["mp4", "mov", "webm", "m4v"]
PAPER_TYPES = ["pdf", "jpg", "jpeg", "png"]

# --------------------------------------------------------------------------- #
# Model registry — same wiring as the other course apps
# --------------------------------------------------------------------------- #
OPENAI_EP = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
FOUNDRY_EP = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
FOUNDRY_KEY = os.environ.get("AZURE_FOUNDRY_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

MODELS = {
    "GPT-5.5": (os.environ.get("MODEL_GPT55_DEPLOYMENT", "gpt-5-5"), OPENAI_EP, OPENAI_KEY),
    "DeepSeek-V4-Pro": (os.environ.get("MODEL_DEEPSEEK_V4_PRO_DEPLOYMENT", "ds-v4pro"), FOUNDRY_EP, FOUNDRY_KEY),
    "Grok-4.3": (os.environ.get("MODEL_GROK43_DEPLOYMENT", "xai-grok43"), FOUNDRY_EP, FOUNDRY_KEY),
    "Mistral-Medium-3.5": (os.environ.get("MODEL_MISTRAL_MEDIUM_35_DEPLOYMENT", "mstr-med35"), FOUNDRY_EP, FOUNDRY_KEY),
}

# The AI tutor model picker has been removed from the UI (per Sharks Academy's
# request) — we just quietly use the first configured model behind the scenes.
DEFAULT_MODEL = next(iter(MODELS))


def ai_ready(model):
    return bool(MODELS[model][1] and MODELS[model][2])


# --------------------------------------------------------------------------- #
# Data helpers (lecture index)
# --------------------------------------------------------------------------- #
def load_index() -> list:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_index(items: list) -> None:
    INDEX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:60]


def add_lecture(title, subject, description, notes, uploaded_file, teacher_id=None) -> None:
    lid = uuid.uuid4().hex[:10]
    ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
    fname = f"{lid}_{safe_name(Path(uploaded_file.name).stem)}{ext}"
    (VIDEO_DIR / fname).write_bytes(uploaded_file.getbuffer())
    items = load_index()
    items.append({
        "id": lid, "title": title.strip(), "subject": subject,
        "description": description.strip(), "notes": notes.strip(),
        "video": fname, "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "teacher_id": teacher_id,
    })
    save_index(items)


def edit_lecture(lid, title, subject, description, notes) -> None:
    """Update the metadata of an existing lecture (video file untouched)."""
    items = load_index()
    for it in items:
        if it["id"] == lid:
            it["title"] = title.strip()
            it["subject"] = subject
            it["description"] = description.strip()
            it["notes"] = notes.strip()
    save_index(items)


def delete_lecture(lid: str) -> None:
    items = load_index()
    for it in items:
        if it["id"] == lid:
            try:
                (VIDEO_DIR / it["video"]).unlink(missing_ok=True)
            except OSError:
                pass
    save_index([it for it in items if it["id"] != lid])


# --------------------------------------------------------------------------- #
# Data helpers (paper checker index)
# --------------------------------------------------------------------------- #
def load_papers() -> list:
    if PAPERS_INDEX_FILE.exists():
        try:
            return json.loads(PAPERS_INDEX_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_papers(items: list) -> None:
    PAPERS_INDEX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def add_paper(title, subject, description, marking_notes, uploaded_file, teacher_id) -> None:
    """Teacher adds a paper (the questions, and optionally a marking scheme in
    `marking_notes`) that students can then upload their solved version against."""
    pid = uuid.uuid4().hex[:10]
    ext = Path(uploaded_file.name).suffix.lower() or ".pdf"
    fname = f"{pid}_{safe_name(Path(uploaded_file.name).stem)}{ext}"
    (PAPER_DIR / fname).write_bytes(uploaded_file.getbuffer())
    items = load_papers()
    items.append({
        "id": pid, "title": title.strip(), "subject": subject,
        "description": description.strip(), "marking_notes": marking_notes.strip(),
        "file": fname, "teacher_id": teacher_id,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_papers(items)


def edit_paper(pid, title, subject, description, marking_notes) -> None:
    items = load_papers()
    for it in items:
        if it["id"] == pid:
            it["title"] = title.strip()
            it["subject"] = subject
            it["description"] = description.strip()
            it["marking_notes"] = marking_notes.strip()
    save_papers(items)


def delete_paper(pid: str) -> None:
    items = load_papers()
    for it in items:
        if it["id"] == pid:
            try:
                (PAPER_DIR / it["file"]).unlink(missing_ok=True)
            except OSError:
                pass
    save_papers([it for it in items if it["id"] != pid])


def get_papers_for_subject(subject) -> list:
    return [p for p in load_papers() if p["subject"] == subject]


def get_papers_for_teacher(teacher_id) -> list:
    return [p for p in load_papers() if p.get("teacher_id") == teacher_id]


def parse_json(text):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None


# --------------------------------------------------------------------------- #
# AI helpers
# --------------------------------------------------------------------------- #
def _call(model, prompt, system, max_tokens=700):
    deployment, endpoint, key = MODELS[model]
    if not endpoint or not key:
        return {"ok": False, "text": f"⚠️ {model}: AI is not configured in this environment."}
    try:
        client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=API_VERSION)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=max_tokens,
        )
        return {"ok": True, "text": resp.choices[0].message.content or ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": f"⚠️ Could not reach {model}: {exc}"}


def _lecture_context(lec) -> str:
    return (f"Lecture: {lec['title']} ({lec['subject']})\n"
            f"Description: {lec.get('description','')}\n"
            f"Notes:\n{lec.get('notes','') or '(no notes provided)'}")


def ask_tutor(model, lec, question):
    system = ("You are a friendly O-Level tutor. Answer the student's question about this "
              "lecture using its notes. If the notes don't cover it, use your general "
              "O-Level knowledge but say so. Keep it clear and simple.")
    return _call(model, f"{_lecture_context(lec)}\n\nSTUDENT QUESTION: {question}",
                 system, max_tokens=600)["text"]


def summarize(model, lec):
    system = "You summarise lessons into clear revision notes for O-Level students."
    prompt = (f"{_lecture_context(lec)}\n\nWrite a revision summary: 5-7 key bullet points "
              "plus one 'exam tip'.")
    return _call(model, prompt, system, max_tokens=600)["text"]


def make_quiz(model, lec, n=5):
    system = "You write clear O-Level multiple-choice questions with one correct answer."
    prompt = (f"{_lecture_context(lec)}\n\nWrite {n} multiple-choice questions testing this "
              "lecture. Return ONLY JSON: "
              '{"questions":[{"q":"...","options":["a","b","c","d"],"answer_index":0,'
              '"explanation":"why"}]}')
    res = _call(model, prompt, system, max_tokens=1100)
    data = parse_json(res["text"]) if res["ok"] else None
    return data.get("questions") if isinstance(data, dict) else None


def generate_practice_questions(model, subject, title, description, mistakes, n=5):
    """Generate fresh practice questions targeting whatever a student got wrong on a
    checked paper, so they can drill the same weak spots without repeating the paper."""
    weak_spots = "; ".join(
        f"{m.get('question', '')}: {m.get('issue', '')}" for m in (mistakes or [])
    ) or "no specific weak spots identified — cover the paper's general topics"
    system = ("You are an O-Level tutor who writes targeted practice questions to help "
              "students drill exactly what they got wrong.")
    prompt = (f"Subject: {subject}\nPaper: {title}\nTopic: {description or '(not given)'}\n"
              f"The student's recent mistakes: {weak_spots}\n\n"
              f"Write {n} new practice questions (mix of multiple-choice and short-answer) "
              "that specifically target these weak spots. Return ONLY JSON: "
              '{"questions":[{"q":"...","answer":"the correct answer or approach"}]}')
    res = _call(model, prompt, system, max_tokens=900)
    data = parse_json(res["text"]) if res["ok"] else None
    return data.get("questions") if isinstance(data, dict) else None


def generate_welcome_note(model, name, subjects):
    """Personalized AI-concierge welcome note shown after sign-up — the app's signature touch."""
    system = ("You are an enthusiastic school orientation concierge who writes short, warm, "
              "personalized welcome notes for new O-Level students. No corporate tone, no "
              "generic filler — make it feel handwritten and specific.")
    prompt = (f"Student name: {name}\n"
              f"Subjects enrolled: {', '.join(subjects) if subjects else 'none yet'}\n\n"
              "Write a short welcome note (3-4 sentences): greet them by name, say something "
              "specific and encouraging about the *combination* of subjects they picked, and "
              "end with one light, motivating line about their first week.")
    res = _call(model, prompt, system, max_tokens=220)
    return res["text"] if res["ok"] else None


def generate_teacher_welcome_note(model, name, subjects):
    """Personalized AI-concierge welcome note shown after a teacher joins/updates their profile."""
    system = ("You are a warm, professional school operations concierge who writes short "
              "welcome notes for teachers joining an O-Level tutoring platform. No corporate "
              "tone, no generic filler — make it feel handwritten and specific.")
    prompt = (f"Teacher name: {name}\n"
              f"Subjects they teach: {', '.join(subjects) if subjects else 'none yet'}\n\n"
              "Write a short welcome note (3-4 sentences): greet them by name, say something "
              "specific and encouraging about the subjects they teach, and end with one line "
              "about how students will be able to find and pick them.")
    res = _call(model, prompt, system, max_tokens=220)
    return res["text"] if res["ok"] else None


@st.cache_data(show_spinner=False)
def video_bytes(path_str, size):
    return Path(path_str).read_bytes()


# --------------------------------------------------------------------------- #
# AI paper checker
# --------------------------------------------------------------------------- #
MAX_PDF_PAGES = 10  # cap how many pages of a PDF we send to the model, to keep cost/latency sane


def _image_bytes_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _file_to_image_data_urls(path: Path) -> list:
    """Turn an uploaded paper/submission (image or PDF) into a list of data-URL
    images the vision model can read. Multi-page PDFs are rendered page by page."""
    ext = path.suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg", "png"):
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return [_image_bytes_to_data_url(path.read_bytes(), mime)]

    if ext == "pdf":
        if not PDF_SUPPORT:
            return []
        urls = []
        doc = fitz.open(path)
        for page in doc[:MAX_PDF_PAGES]:
            pix = page.get_pixmap(dpi=150)
            urls.append(_image_bytes_to_data_url(pix.tobytes("png")))
        doc.close()
        return urls

    return []


def _save_uploaded_temp(uploaded_file) -> Path:
    ext = Path(uploaded_file.name).suffix.lower() or ".png"
    fname = f"{uuid.uuid4().hex[:10]}{ext}"
    path = SUBMISSION_DIR / fname
    path.write_bytes(uploaded_file.getbuffer())
    return path


def check_paper(model, title, subject, description="", marking_notes="",
                reference_path=None, reference_upload=None,
                submission_file=None, submission_text=None):
    """Ask the AI to mark a student's solved paper.

    `reference_path` (a Path already on disk, e.g. a teacher-added paper) and
    `reference_upload` (a freshly uploaded file, e.g. a student's own copy of
    the paper) are both optional — a student can check a paper with no
    reference at all, in which case the AI marks using general subject
    knowledge. Returns a dict with the parsed result, or
    {"ok": False, "text": <error>} if something went wrong.
    """
    deployment, endpoint, key = MODELS[model]
    if not endpoint or not key:
        return {"ok": False, "text": f"⚠️ {model}: AI is not configured in this environment."}
    if not submission_file and not (submission_text or "").strip():
        return {"ok": False, "text": "Please upload your solved paper or type your answers."}

    marking_display = marking_notes or (
        "none provided — use your general O-Level marking standards for this subject"
    )
    prompt_text = (
        f"You are an O-Level examiner. Here is context about the paper being submitted.\n"
        f"Title: {title}\n"
        f"Subject: {subject}\n"
        f"Description: {description or '(none given)'}\n"
        f"Marking scheme / notes: {marking_display}\n\n"
        "If reference paper images are provided below, they show the original questions "
        "(and possibly an answer key) — use them to mark accurately. If no reference is "
        "provided, mark using your own general subject knowledge. Then the student's solved "
        "submission follows — mark it question by question.\n\n"
        "Return ONLY JSON in this exact shape:\n"
        '{"score_percent": 0-100, "overall_feedback": "2-3 sentences", '
        '"strengths": ["..."], '
        '"mistakes": [{"question": "...", "issue": "...", "correction": "..."}]}'
    )
    content = [{"type": "text", "text": prompt_text}]

    ref_path = reference_path
    if reference_upload is not None:
        ref_path = _save_uploaded_temp(reference_upload)

    if ref_path is not None and ref_path.exists():
        is_ref_pdf = ref_path.suffix.lower() == ".pdf"
        if is_ref_pdf and not PDF_SUPPORT:
            content.append({"type": "text", "text": "(A reference PDF was provided but PDF "
                            "support isn't installed on this server, so it was skipped.)"})
        else:
            ref_urls = _file_to_image_data_urls(ref_path)
            if ref_urls:
                content.append({"type": "text", "text": "--- REFERENCE PAPER (questions/answer key) ---"})
                for url in ref_urls:
                    content.append({"type": "image_url", "image_url": {"url": url}})

    content.append({"type": "text", "text": "--- STUDENT SUBMISSION ---"})
    if submission_file is not None:
        sub_path = _save_uploaded_temp(submission_file)
        is_pdf = sub_path.suffix.lower() == ".pdf"
        if is_pdf and not PDF_SUPPORT:
            return {"ok": False, "text": "⚠️ PDF support isn't installed in this environment "
                                         "(`pip install pymupdf`). Please upload a JPG/PNG "
                                         "instead."}
        urls = _file_to_image_data_urls(sub_path)
        if not urls:
            return {"ok": False, "text": "Couldn't read that file. Please upload a PDF, JPG, or PNG."}
        if is_pdf:
            doc = fitz.open(sub_path)
            page_count = doc.page_count
            doc.close()
            if page_count > MAX_PDF_PAGES:
                content.append({"type": "text", "text": f"(Note: only the first "
                                f"{MAX_PDF_PAGES} of {page_count} pages were included.)"})
        for url in urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content.append({"type": "text", "text": submission_text.strip()})

    try:
        client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=API_VERSION)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a precise, fair O-Level examiner. "
                                              "Always reply with the requested JSON only."},
                {"role": "user", "content": content},
            ],
            temperature=1,
            max_completion_tokens=1400,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": f"⚠️ Could not reach {model}: {exc}"}

    parsed = parse_json(text)
    if not isinstance(parsed, dict):
        return {"ok": False, "text": "The AI's response couldn't be read. Please try again."}
    return {"ok": True, **parsed}


# --------------------------------------------------------------------------- #
# Preference-system database
# --------------------------------------------------------------------------- #
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_tables():
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            grade_level TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT UNIQUE NOT NULL,
            subject_code TEXT UNIQUE
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT,
            max_students INTEGER DEFAULT 30
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_subjects (
            teacher_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            PRIMARY KEY (teacher_id, subject_id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            preferred_teacher_id INTEGER,
            priority INTEGER DEFAULT 1,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
            FOREIGN KEY (preferred_teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL,
            UNIQUE (student_id, subject_id, priority)
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lecture_progress (
            student_id INTEGER NOT NULL,
            lecture_id TEXT NOT NULL,
            status TEXT DEFAULT 'in_progress',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (student_id, lecture_id),
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            paper_id TEXT,
            title TEXT,
            subject TEXT,
            score_percent REAL,
            source TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


def _ensure_column(table, column, coltype_and_default):
    """Add `column` to `table` if it doesn't already exist (simple SQLite migration)."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table});")
    existing = [row[1] for row in cursor.fetchall()]
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype_and_default};")
        conn.commit()
    conn.close()


def seed_subjects():
    """Keep the `subjects` table in sync with the app's SUBJECTS list."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO subjects (subject_name) VALUES (?);",
        [(s,) for s in SUBJECTS]
    )
    conn.commit()
    conn.close()


def get_student_by_roll(roll_number):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, full_name, grade_level, email FROM students WHERE roll_number = ?;",
                   (roll_number,))
    row = cursor.fetchone()
    conn.close()
    return row  # (student_id, full_name, grade_level, email) or None


def get_student_by_id(student_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, full_name, roll_number, grade_level, email FROM students WHERE student_id = ?;",
                   (student_id,))
    row = cursor.fetchone()
    conn.close()
    return row  # (student_id, full_name, roll_number, grade_level, email) or None


def delete_student(student_id) -> None:
    """Permanently delete a student's profile. Their subject/teacher preferences are
    removed automatically via ON DELETE CASCADE."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE student_id = ?;", (student_id,))
    conn.commit()
    conn.close()


def generate_unique_student_id() -> str:
    """Auto-assign a unique Student ID like OL-2026-0001, instead of letting students
    pick their own (which is how two students could end up clashing)."""
    year = datetime.now().year
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students WHERE roll_number LIKE ?;", (f"OL-{year}-%",))
    count = cursor.fetchone()[0]
    conn.close()
    candidate = f"OL-{year}-{count + 1:04d}"
    # Guard against the rare case of two students signing up at almost the same
    # instant and colliding on the same number.
    while get_student_by_roll(candidate):
        count += 1
        candidate = f"OL-{year}-{count + 1:04d}"
    return candidate


def register_student(full_name, roll_number, grade_level, email):
    """Add a new student. Returns the student_id, or None if roll_number already exists."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (full_name, roll_number, grade_level, email) VALUES (?, ?, ?, ?);",
            (full_name, roll_number, grade_level, email)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def register_teacher_for_subjects(full_name, email, subject_names):
    """Create the teacher if needed, then link them to each chosen subject."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT teacher_id FROM teachers WHERE full_name = ?;", (full_name,))
    row = cursor.fetchone()
    if row:
        teacher_id = row[0]
        if email:
            cursor.execute("UPDATE teachers SET email = ? WHERE teacher_id = ?;", (email, teacher_id))
    else:
        cursor.execute("INSERT INTO teachers (full_name, email) VALUES (?, ?);", (full_name, email))
        teacher_id = cursor.lastrowid

    for subject_name in subject_names:
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
        srow = cursor.fetchone()
        if srow:
            cursor.execute(
                "INSERT OR IGNORE INTO teacher_subjects (teacher_id, subject_id) VALUES (?, ?);",
                (teacher_id, srow[0])
            )

    conn.commit()
    conn.close()
    return teacher_id


def set_teacher_subjects(teacher_id, subject_names):
    """Replace a teacher's subject links entirely with the given list."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM teacher_subjects WHERE teacher_id = ?;", (teacher_id,))
    for subject_name in subject_names:
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
        srow = cursor.fetchone()
        if srow:
            cursor.execute(
                "INSERT OR IGNORE INTO teacher_subjects (teacher_id, subject_id) VALUES (?, ?);",
                (teacher_id, srow[0])
            )
    conn.commit()
    conn.close()


def get_teacher_by_name(full_name):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT teacher_id, full_name, email FROM teachers WHERE full_name = ?;", (full_name,))
    row = cursor.fetchone()
    conn.close()
    return row  # (teacher_id, full_name, email) or None


def get_teacher_by_id(teacher_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT teacher_id, full_name, email FROM teachers WHERE teacher_id = ?;", (teacher_id,))
    row = cursor.fetchone()
    conn.close()
    return row  # (teacher_id, full_name, email) or None


def delete_teacher(teacher_id) -> None:
    """Permanently delete a teacher's profile. Their subject links are removed via
    ON DELETE CASCADE; any student preferences pointing to them fall back to
    'No preference' via ON DELETE SET NULL. Lectures/papers they added are left in
    place (not deleted) so that content isn't lost."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM teachers WHERE teacher_id = ?;", (teacher_id,))
    conn.commit()
    conn.close()


def get_subjects_for_teacher(teacher_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.subject_name
        FROM teacher_subjects ts
        JOIN subjects s ON s.subject_id = ts.subject_id
        WHERE ts.teacher_id = ?
        ORDER BY s.subject_name;
    """, (teacher_id,))
    rows = [r[0] for r in cursor.fetchall()]
    conn.close()
    return rows


def get_teachers_for_subject(subject_name):
    """List teachers who teach a given subject."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.teacher_id, t.full_name
        FROM teachers t
        JOIN teacher_subjects ts ON ts.teacher_id = t.teacher_id
        JOIN subjects s ON s.subject_id = ts.subject_id
        WHERE s.subject_name = ?
        ORDER BY t.full_name;
    """, (subject_name,))
    rows = cursor.fetchall()
    conn.close()
    return rows  # list of (teacher_id, full_name)


def submit_preference(student_id, subject_name, preferred_teacher_id=None, priority=1):
    """Record (or update) a student's subject + preferred teacher choice."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return False, f"Subject '{subject_name}' not found."
    subject_id = row[0]

    try:
        cursor.execute("""
            INSERT INTO student_preferences (student_id, subject_id, preferred_teacher_id, priority)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id, subject_id, priority)
            DO UPDATE SET preferred_teacher_id = excluded.preferred_teacher_id,
                          submitted_at = CURRENT_TIMESTAMP;
        """, (student_id, subject_id, preferred_teacher_id, priority))
        conn.commit()
        return True, "Preference saved."
    except sqlite3.IntegrityError as exc:
        return False, str(exc)
    finally:
        conn.close()


def view_student_preferences(student_id):
    """Return all preferences for a student, in priority order."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sub.subject_name, t.full_name, sp.priority
        FROM student_preferences sp
        JOIN subjects sub ON sub.subject_id = sp.subject_id
        LEFT JOIN teachers t ON t.teacher_id = sp.preferred_teacher_id
        WHERE sp.student_id = ?
        ORDER BY sp.priority ASC;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows  # list of (subject_name, teacher_name_or_None, priority)


def get_preferences_for_teacher(teacher_id):
    """Which students picked this teacher, and for which subject."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.full_name, s.roll_number, sub.subject_name, sp.priority
        FROM student_preferences sp
        JOIN students s ON s.student_id = sp.student_id
        JOIN subjects sub ON sub.subject_id = sp.subject_id
        WHERE sp.preferred_teacher_id = ?
        ORDER BY sub.subject_name, sp.priority;
    """, (teacher_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# --------------------------------------------------------------------------- #
# Lecture completion tracking
# --------------------------------------------------------------------------- #
def mark_lecture_progress(student_id, lecture_id, status="done"):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO lecture_progress (student_id, lecture_id, status, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(student_id, lecture_id)
        DO UPDATE SET status = excluded.status, updated_at = CURRENT_TIMESTAMP;
    """, (student_id, lecture_id, status))
    conn.commit()
    conn.close()


def get_lecture_progress(student_id) -> dict:
    """Returns {lecture_id: status} for a student."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT lecture_id, status FROM lecture_progress WHERE student_id = ?;",
                   (student_id,))
    rows = dict(cursor.fetchall())
    conn.close()
    return rows


def get_subject_progress(student_id, subject_lectures) -> tuple:
    """subject_lectures: list of lecture ids in a subject. Returns (done_count, total_count)."""
    if not subject_lectures:
        return 0, 0
    progress = get_lecture_progress(student_id)
    done = sum(1 for lid in subject_lectures if progress.get(lid) == "done")
    return done, len(subject_lectures)


# --------------------------------------------------------------------------- #
# Paper-checker result history
# --------------------------------------------------------------------------- #
def record_paper_result(student_id, paper_id, title, subject, score_percent, source):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO paper_results (student_id, paper_id, title, subject, score_percent, source)
        VALUES (?, ?, ?, ?, ?, ?);
    """, (student_id, paper_id, title, subject, score_percent, source))
    conn.commit()
    result_id = cursor.lastrowid
    conn.close()
    return result_id


def get_results_for_student(student_id) -> list:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT result_id, title, subject, score_percent, source, checked_at
        FROM paper_results WHERE student_id = ? ORDER BY checked_at DESC;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


# --------------------------------------------------------------------------- #
# UI — Sharks Academy branding
# --------------------------------------------------------------------------- #
_ORG_HEADER_CSS = """
<style>
.sharks-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #00B894 100%);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 34px rgba(15,32,39,0.35);
    text-align: center;
}
.sharks-header h1 {
    margin: 0;
    font-size: 2.5rem;
    letter-spacing: .5px;
    font-weight: 800;
}
.sharks-header p {
    margin-top: .5rem;
    opacity: .88;
    font-size: 1.05rem;
}

/* --- Role selector (Student / Teacher) --- */
.role-label {
    font-size: 1rem;
    font-weight: 700;
    color: #1e1e2f;
    margin-bottom: .5rem;
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button {
    width: 100%;
    border-radius: 16px !important;
    padding: 1.1rem .6rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin-bottom: .9rem !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button:hover {
    transform: translateY(-2px);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7, #00B894) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(108,92,231,.40);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: #f4f3fb !important;
    color: #3d3d54 !important;
    border: 2px solid #e2dff5 !important;
}

/* --- Sidebar infographic (Sharks Academy at a glance) --- */
.sidebar-infographic-title {
    font-size: .95rem; font-weight: 800; color: #1e1e2f;
    margin: .2rem 0 .7rem 0; display: flex; align-items: center; gap: .4rem;
}
.sidebar-stat-card {
    display: flex; align-items: center; gap: .7rem;
    background: linear-gradient(135deg, #6C5CE7 0%, #00B894 100%);
    border-radius: 12px; padding: .7rem .9rem; margin-bottom: .55rem;
    color: white; box-shadow: 0 4px 12px rgba(108,92,231,.22);
    transition: transform .15s ease, box-shadow .15s ease;
}
.sidebar-stat-card:hover {
    transform: translateX(5px) scale(1.02);
    box-shadow: 0 8px 20px rgba(108,92,231,.38);
}
.sidebar-stat-icon {
    font-size: 1.5rem; flex-shrink: 0;
    background: rgba(255,255,255,.18); border-radius: 10px;
    width: 2.2rem; height: 2.2rem; display: flex; align-items: center; justify-content: center;
}
.sidebar-stat-value { font-size: .95rem; font-weight: 800; line-height: 1.2; }
.sidebar-stat-label { font-size: .72rem; opacity: .88; font-weight: 500; }
</style>
"""


def _render_org_header():
    st.markdown(_ORG_HEADER_CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="sharks-header">
        <h1>🦈 Sharks Academy</h1>
        <p>Empowering every student, one lecture at a time.</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_sidebar_infographic():
    """A quick, eye-catching 'Sharks Academy at a glance' stat block in the sidebar,
    right where students/teachers choose their role."""
    stats = [
        ("🎓", "1000+", "Students educated"),
        ("👩‍🏫", "Qualified", "Teachers"),
        ("📚", "12+", "Subjects covered"),
    ]
    cards = "".join(
        f'<div class="sidebar-stat-card">'
        f'<span class="sidebar-stat-icon">{icon}</span>'
        f'<div><div class="sidebar-stat-value">{value}</div>'
        f'<div class="sidebar-stat-label">{label}</div></div>'
        f'</div>'
        for icon, value, label in stats
    )
    st.sidebar.markdown(
        f'<div class="sidebar-infographic-title">✨ Sharks Academy at a glance</div>{cards}',
        unsafe_allow_html=True,
    )


def _render_role_selector():
    st.sidebar.markdown('<div class="role-label">👋 I am a…</div>', unsafe_allow_html=True)
    st.session_state.setdefault("role", "Student")

    c1, c2 = st.sidebar.columns(2)
    if c1.button("🎓 Student", key="role_student",
                 type="primary" if st.session_state.role == "Student" else "secondary",
                 use_container_width=True):
        st.session_state.role = "Student"
        st.rerun()
    if c2.button("👩‍🏫 Teacher", key="role_teacher",
                 type="primary" if st.session_state.role == "Teacher" else "secondary",
                 use_container_width=True):
        st.session_state.role = "Teacher"
        st.rerun()

    return st.session_state.role


# --------------------------------------------------------------------------- #
# UI — Shared "hero + step wizard" styling (used by both student sign-up and
# the teacher profile wizard)
# --------------------------------------------------------------------------- #
_HERO_CSS = """
<style>
.sh-hero {
    background: linear-gradient(135deg, #6C5CE7 0%, #00B894 100%);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(108,92,231,0.30);
}
.sh-hero.sh-hero-teacher {
    background: linear-gradient(135deg, #E17055 0%, #6C5CE7 100%);
    box-shadow: 0 10px 30px rgba(225,112,85,0.28);
}
.sh-hero h1 { margin: 0; font-size: 1.9rem; }
.sh-hero p { opacity: .92; margin-top: .5rem; font-size: 1rem; }
.sh-step-pill {
    display: inline-block; padding: .32rem 1rem; border-radius: 999px;
    font-size: .8rem; font-weight: 600; margin-right: .4rem; margin-bottom: .6rem;
}
.sh-step-active { background: #6C5CE7; color: white; }
.sh-step-done   { background: #00B894; color: white; }
.sh-step-todo   { background: #eef0f5; color: #8a8f9c; }
.sh-id-card {
    background: linear-gradient(135deg, #1e1e2f, #2d2d44);
    border-radius: 18px; padding: 1.6rem 1.7rem; color: white;
    max-width: 460px; box-shadow: 0 8px 26px rgba(0,0,0,.28);
}
.sh-id-avatar {
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(135deg, #6C5CE7, #00B894);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.25rem; margin-bottom: .7rem;
}
.sh-id-name { font-size: 1.15rem; font-weight: 700; }
.sh-id-meta { opacity: .7; font-size: .82rem; margin-bottom: .7rem; }
.sh-subject-badge {
    display: inline-block; background: rgba(255,255,255,.12);
    padding: .28rem .75rem; border-radius: 999px; font-size: .78rem;
    margin: .2rem .3rem .2rem 0;
}
.sh-note {
    border-left: 4px solid #6C5CE7; background: rgba(108,92,231,.06);
    padding: .9rem 1.1rem; border-radius: 8px; margin-top: .8rem;
}
.sh-lecture-card {
    border: 1px solid #ece9f7; border-radius: 14px; padding: 1rem 1.2rem;
    margin-bottom: .8rem; background: #fbfaff;
}

/* --- Teacher stat cards + subject box, coloured to match the student hero --- */
.ol-stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: .9rem;
    margin-bottom: 1.3rem;
}
.ol-stat-card {
    background: linear-gradient(135deg, #6C5CE7 0%, #00B894 100%);
    border-radius: 14px; padding: 1.05rem 1.25rem; position: relative; color: white;
    box-shadow: 0 6px 18px rgba(108,92,231,.22);
}
.ol-stat-icon { position: absolute; top: 1.05rem; right: 1.1rem; font-size: 1.3rem; opacity: .9; }
.ol-stat-label { color: rgba(255,255,255,.85); font-size: .8rem; margin-bottom: .3rem; }
.ol-stat-value { font-size: 1.5rem; font-weight: 800; color: white; }
@media (max-width: 900px) {
    .ol-stat-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""


def _inject_hero_css():
    st.markdown(_HERO_CSS, unsafe_allow_html=True)


def _render_steps(current, labels):
    html = ""
    for i, label in enumerate(labels, start=1):
        if i == current:
            cls = "sh-step-active"
        elif i < current:
            cls = "sh-step-done"
        else:
            cls = "sh-step-todo"
        html += f'<span class="sh-step-pill {cls}">{label}</span>'
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# UI — Student: lecture playback
# --------------------------------------------------------------------------- #
def _play_lecture(lec, model, student_id):
    st.markdown(f"### {lec['title']}")
    if lec.get("description"):
        st.caption(lec["description"])
    path = VIDEO_DIR / lec["video"]
    if path.exists():
        st.video(video_bytes(str(path), path.stat().st_size))
    else:
        st.error("Video file is missing (it may have been reset on redeploy).")

    progress = get_lecture_progress(student_id)
    is_done = progress.get(lec["id"]) == "done"
    c1, c2 = st.columns([1, 4])
    if is_done:
        c1.success("✅ Watched")
        if c2.button("Mark as not watched", key=f"unwatch_{lec['id']}"):
            mark_lecture_progress(student_id, lec["id"], "in_progress")
            st.rerun()
    else:
        if c1.button("✅ Mark as watched", key=f"watch_{lec['id']}", type="primary"):
            mark_lecture_progress(student_id, lec["id"], "done")
            st.rerun()

    tab_notes, tab_ask, tab_quiz = st.tabs(["📄 Notes & summary", "💬 Ask the tutor", "🧠 Quiz me"])

    with tab_notes:
        st.markdown(lec.get("notes") or "_No notes were added for this lecture._")
        if ai_ready(model) and lec.get("notes"):
            if st.button("✨ Summarise for revision", key=f"sum_{lec['id']}"):
                with st.spinner("Summarising…"):
                    st.session_state[f"summary_{lec['id']}"] = summarize(model, lec)
            if st.session_state.get(f"summary_{lec['id']}"):
                st.info(st.session_state[f"summary_{lec['id']}"])

    with tab_ask:
        if not ai_ready(model):
            st.info("The AI tutor isn't configured in this environment.")
        else:
            q = st.text_input("Ask anything about this lecture",
                              key=f"q_{lec['id']}", placeholder="e.g. Why is chlorophyll important?")
            if st.button("Ask", key=f"ask_{lec['id']}", type="primary") and q.strip():
                with st.spinner("Thinking…"):
                    st.session_state[f"ans_{lec['id']}"] = ask_tutor(model, lec, q)
            if st.session_state.get(f"ans_{lec['id']}"):
                st.markdown(st.session_state[f"ans_{lec['id']}"])

    with tab_quiz:
        if not ai_ready(model):
            st.info("Quizzes need the AI, which isn't configured here.")
        else:
            _quiz_ui(lec, model)


def _quiz_ui(lec, model):
    qkey = f"quiz_{lec['id']}"
    if st.button("🎯 Make me a quiz", key=f"mkquiz_{lec['id']}"):
        with st.spinner("Writing your quiz…"):
            st.session_state[qkey] = make_quiz(model, lec)
            st.session_state[f"{qkey}_submitted"] = False
    quiz = st.session_state.get(qkey)
    if not quiz:
        return
    answers = {}
    for i, item in enumerate(quiz):
        st.markdown(f"**Q{i+1}. {item['q']}**")
        answers[i] = st.radio("Pick one", item["options"], index=None,
                              key=f"{qkey}_{i}", label_visibility="collapsed")
    if st.button("Submit answers", key=f"{qkey}_submit", type="primary"):
        st.session_state[f"{qkey}_submitted"] = True
    if st.session_state.get(f"{qkey}_submitted"):
        correct = 0
        for i, item in enumerate(quiz):
            chosen = answers.get(i)
            right = item["options"][item["answer_index"]]
            ok = chosen == right
            correct += int(ok)
            st.markdown(("✅" if ok else "❌") + f" **Q{i+1}** — correct: *{right}*")
            st.caption("💡 " + item.get("explanation", ""))
        st.markdown(f"### Score: {correct}/{len(quiz)}")
        if correct == len(quiz):
            st.balloons()


# --------------------------------------------------------------------------- #
# UI — Student: AI paper checker
# --------------------------------------------------------------------------- #
def _render_paper_result(result, model, context, student_id):
    score = result.get("score_percent")
    if isinstance(score, (int, float)):
        st.metric("Score", f"{score:.0f}%" if isinstance(score, float) else f"{score}%")
        st.progress(min(max(int(score), 0), 100) / 100)

    if result.get("overall_feedback"):
        st.info(result["overall_feedback"])

    strengths = result.get("strengths") or []
    if strengths:
        st.markdown("#### ✅ What you did well")
        for s in strengths:
            st.write(f"- {s}")

    mistakes = result.get("mistakes") or []
    if mistakes:
        st.markdown("#### 🔍 Mistakes & corrections")
        for m in mistakes:
            with st.expander(f"❌ {m.get('question', 'Question')}"):
                st.write(f"**Issue:** {m.get('issue', '')}")
                st.write(f"**Correction:** {m.get('correction', '')}")
    else:
        st.success("No mistakes found — great work! 🎉")

    st.divider()
    practice_key = f"practice_{context['key']}"
    if ai_ready(model) and st.button("🎯 Generate 5 practice questions on my weak spots",
                                     key=f"genpractice_{context['key']}"):
        with st.spinner("Writing targeted practice questions…"):
            st.session_state[practice_key] = generate_practice_questions(
                model, context["subject"], context["title"], context["description"], mistakes)
    practice_qs = st.session_state.get(practice_key)
    if practice_qs:
        st.markdown("#### 🎯 Practice questions for you")
        for i, q in enumerate(practice_qs, start=1):
            with st.expander(f"Q{i}. {q.get('q', '')}"):
                st.write(f"**Answer:** {q.get('answer', '')}")


def _paper_checker_teacher_mode(model, student_id):
    items = load_papers()
    if not items:
        st.info("📭 No papers from teachers yet. Try **🙋 Check my own paper** instead — "
                "you don't need a teacher to use the checker.")
        return

    subjects = sorted({it["subject"] for it in items})
    subject = st.selectbox("📚 Choose a subject", subjects, key="paper_check_subject")
    in_subject = [it for it in items if it["subject"] == subject]
    titles = [it["title"] for it in in_subject]
    picked = st.selectbox("📝 Choose a paper", range(len(in_subject)),
                          format_func=lambda i: titles[i], key="paper_check_pick")
    paper = in_subject[picked]

    if paper.get("description"):
        st.caption(paper["description"])
    paper_path = PAPER_DIR / paper["file"]
    if paper_path.exists():
        if paper_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            st.image(str(paper_path), caption="Paper questions", width=400)
        else:
            st.download_button("📄 Download the paper", data=paper_path.read_bytes(),
                               file_name=paper_path.name, key=f"dl_{paper['id']}")

    st.divider()
    st.markdown("#### Submit your solved paper")
    mode = st.radio("How would you like to submit?", ["Upload a file", "Type my answers"],
                    horizontal=True, key=f"mode_{paper['id']}")

    submission_file, submission_text = None, None
    if mode == "Upload a file":
        submission_file = st.file_uploader("Your solved paper (photo or PDF)", type=PAPER_TYPES,
                                           key=f"sub_file_{paper['id']}")
    else:
        submission_text = st.text_area("Type your answers here", height=200,
                                       key=f"sub_text_{paper['id']}")

    if st.button("✅ Check my paper", type="primary", key=f"checkbtn_{paper['id']}"):
        with st.spinner("Marking your paper…"):
            result = check_paper(model, paper["title"], paper["subject"],
                                 description=paper.get("description", ""),
                                 marking_notes=paper.get("marking_notes", ""),
                                 reference_path=paper_path,
                                 submission_file=submission_file,
                                 submission_text=submission_text)
        if not result.get("ok"):
            st.error(result.get("text", "Something went wrong."))
        else:
            st.session_state[f"paper_result_{paper['id']}"] = result
            if isinstance(result.get("score_percent"), (int, float)):
                record_paper_result(student_id, paper["id"], paper["title"], paper["subject"],
                                    result["score_percent"], "teacher")

    stored = st.session_state.get(f"paper_result_{paper['id']}")
    if stored:
        st.divider()
        context = {"key": paper["id"], "subject": paper["subject"], "title": paper["title"],
                  "description": paper.get("description", "")}
        _render_paper_result(stored, model, context, student_id)


def _paper_checker_solo_mode(model, student_id):
    st.caption("No teacher needed — fill in a few details, submit your solved paper, and "
              "the AI will mark it using your reference (if you have one) or its own "
              "subject knowledge.")

    subject = st.selectbox("📚 Subject", SUBJECTS, key="solo_subject")
    title = st.text_input("Paper title (optional)", placeholder="e.g. Practice worksheet 3",
                          key="solo_title")
    description = st.text_input("What's this paper about? (optional)",
                                placeholder="e.g. Quadratic equations, chapter 5",
                                key="solo_description")

    with st.expander("➕ Have the original paper or an answer key? (optional — improves accuracy)"):
        reference_upload = st.file_uploader("Reference paper / answer key", type=PAPER_TYPES,
                                            key="solo_ref_upload")
        marking_notes = st.text_area("Or just paste the answer key / key points here",
                                     height=120, key="solo_marking_notes")

    st.divider()
    st.markdown("#### Submit your solved paper")
    mode = st.radio("How would you like to submit?", ["Upload a file", "Type my answers"],
                    horizontal=True, key="solo_submit_mode")

    submission_file, submission_text = None, None
    if mode == "Upload a file":
        submission_file = st.file_uploader("Your solved paper (photo or PDF)", type=PAPER_TYPES,
                                           key="solo_sub_file")
    else:
        submission_text = st.text_area("Type your answers here", height=200, key="solo_sub_text")

    if st.button("✅ Check my paper", type="primary", key="solo_checkbtn"):
        final_title = title.strip() or "My paper"
        with st.spinner("Marking your paper…"):
            result = check_paper(model, final_title, subject,
                                 description=description, marking_notes=marking_notes,
                                 reference_upload=reference_upload,
                                 submission_file=submission_file,
                                 submission_text=submission_text)
        if not result.get("ok"):
            st.error(result.get("text", "Something went wrong."))
        else:
            st.session_state["solo_paper_result"] = result
            if isinstance(result.get("score_percent"), (int, float)):
                record_paper_result(student_id, None, final_title, subject,
                                    result["score_percent"], "solo")

    stored = st.session_state.get("solo_paper_result")
    if stored:
        st.divider()
        context = {"key": "solo", "subject": subject,
                  "title": title.strip() or "My paper", "description": description}
        _render_paper_result(stored, model, context, student_id)


def _paper_checker_ui(model, student_id):
    st.subheader("📝 AI Paper Checker")
    st.caption("Get instant, question-by-question AI feedback on a solved paper — either one "
              "your teacher added, or entirely on your own.")

    if not ai_ready(model):
        st.info("The AI paper checker isn't configured in this environment.")
        return

    mode = st.radio("How do you want to check a paper?",
                    ["📋 Check a teacher's paper", "🙋 Check my own paper (no teacher needed)"],
                    key="paper_checker_top_mode")

    st.write("")
    if mode.startswith("📋"):
        _paper_checker_teacher_mode(model, student_id)
    else:
        _paper_checker_solo_mode(model, student_id)

    history = get_results_for_student(student_id)
    if history:
        with st.expander(f"📊 My paper-checker history ({len(history)})"):
            for result_id, h_title, h_subject, h_score, h_source, h_checked in history[:20]:
                score_display = f"{h_score:.0f}%" if h_score is not None else "—"
                st.write(f"**{h_title}** · {h_subject} · {score_display} · "
                        f"{'own paper' if h_source == 'solo' else 'teacher paper'} · {h_checked}")


# --------------------------------------------------------------------------- #
# UI — Student sign-up wizard
#
# A branded, 3-step onboarding flow: details -> subjects & teachers -> a
# generated "digital enrollment card" + an AI concierge welcome note.
# --------------------------------------------------------------------------- #
_STEP_LABELS = ["1 · Your details", "2 · Subjects & teachers", "3 · Confirm"]


def _render_hero():
    st.markdown(
        """<div class="sh-hero">
        <h1>🚀 Join the Study Hub</h1>
        <p>Set up your profile once — pick your subjects, choose the teachers you vibe with,
        and get a personalized welcome note from our AI concierge.</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _init_signup_state():
    st.session_state.setdefault("signup_step", 1)
    st.session_state.setdefault("signup_data", {})
    st.session_state.setdefault("editing_prefs", False)


def _step_details():
    st.subheader("Step 1 — Tell us about you")

    with st.expander("🔑 Already have a Student ID? Continue where you left off"):
        existing_id = st.text_input("Student ID", placeholder="e.g. OL-2026-0001",
                                    key="existing_student_id_lookup")
        if existing_id.strip():
            existing = get_student_by_roll(existing_id.strip())
            if existing:
                st.info(f"Welcome back, **{existing[1]}**!")
                if st.button("Continue as this student →", type="primary", key="continue_existing_student"):
                    st.session_state.student_id = existing[0]
                    st.session_state.student_name = existing[1]
                    st.rerun()
            else:
                st.warning("We couldn't find that Student ID — double check it, or fill in "
                          "the form below to sign up as a new student.")

    st.markdown("#### New here? We'll assign you a Student ID once you're done")
    name = st.text_input("Full name", value=st.session_state.signup_data.get("name", ""))
    email = st.text_input("Email address", value=st.session_state.signup_data.get("email", ""),
                          placeholder="you@example.com")
    grade = st.text_input("Grade level", value=st.session_state.signup_data.get("grade", ""),
                          placeholder="e.g. O Level Year 2")

    _, c2 = st.columns([1, 1])
    if c2.button("Next: Subjects →", type="primary"):
        if not (name.strip() and email.strip()):
            st.warning("Name and email are required.")
        elif "@" not in email:
            st.warning("Please enter a valid email address.")
        else:
            st.session_state.signup_data.update({
                "name": name.strip(), "email": email.strip(), "grade": grade.strip(),
            })
            st.session_state.signup_step = 2
            st.rerun()


def _step_subjects():
    st.subheader("Step 2 — Pick your subjects & preferred teachers")
    st.caption("Choose every subject you're taking. For each one you can optionally pick "
              "the teacher you'd like, and rank how important that pick is to you.")

    existing_map = st.session_state.signup_data.get("subjects", {})
    chosen = st.multiselect("Which subjects are you taking?", SUBJECTS,
                            default=list(existing_map.keys()))

    new_map = {}
    for subj in chosen:
        teachers = get_teachers_for_subject(subj)
        names = ["No preference"] + [t[1] for t in teachers]
        prev = existing_map.get(subj, {})
        default_idx = names.index(prev["teacher_name"]) if prev.get("teacher_name") in names else 0

        col1, col2 = st.columns([2, 1])
        with col1:
            t_choice = st.selectbox(f"Preferred teacher — {subj}", names,
                                    index=default_idx, key=f"tch_{subj}")
        with col2:
            pr = st.number_input("Priority", min_value=1, max_value=5,
                                 value=prev.get("priority", 1), step=1, key=f"pr_{subj}")

        teacher_id = None
        if t_choice != "No preference":
            teacher_id = next(t[0] for t in teachers if t[1] == t_choice)
        new_map[subj] = {"teacher_id": teacher_id, "teacher_name": t_choice, "priority": int(pr)}

    c1, c2 = st.columns([1, 1])
    if c1.button("← Back"):
        if st.session_state.editing_prefs:
            st.session_state.editing_prefs = False
        else:
            st.session_state.signup_step = 1
        st.rerun()
    if c2.button("Next: Review →", type="primary"):
        if not chosen:
            st.warning("Pick at least one subject.")
        else:
            st.session_state.signup_data["subjects"] = new_map
            st.session_state.signup_step = 3
            st.rerun()


def _step_confirm(model):
    st.subheader("Step 3 — Review & confirm")
    data = st.session_state.signup_data
    subjects = data.get("subjects", {})

    # Assign a Student ID up front (only for a brand-new sign-up) so it's visible in
    # the preview card below and stays stable if the student navigates back and forth.
    if not st.session_state.get("student_id") and not data.get("roll"):
        data["roll"] = generate_unique_student_id()

    initials = "".join(p[0].upper() for p in (data.get("name") or "?").split()[:2]) or "?"
    badges = "".join(
        f'<span class="sh-subject-badge">{s} · {v["teacher_name"]}</span>'
        for s, v in subjects.items()
    )
    st.markdown(f"""
    <div class="sh-id-card">
      <div class="sh-id-avatar">{initials}</div>
      <div class="sh-id-name">{data.get('name', '')}</div>
      <div class="sh-id-meta">{data.get('roll', '')} · {data.get('grade') or '—'} · {data.get('email', '')}</div>
      <div>{badges}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1, 1])
    if c1.button("← Back"):
        st.session_state.signup_step = 2
        st.rerun()
    if c2.button("✅ Confirm & join", type="primary"):
        if st.session_state.get("student_id"):
            student_id = st.session_state.student_id
        else:
            # Try the ID generated above; on the extremely rare chance it was just
            # taken by someone else, regenerate a fresh one and retry — never fall
            # back to logging this student into an existing account.
            student_id = None
            for _ in range(5):
                student_id = register_student(data["name"], data["roll"], data.get("grade", ""), data["email"])
                if student_id is not None:
                    break
                data["roll"] = generate_unique_student_id()
            if student_id is None:
                st.error("Couldn't assign a Student ID right now — please try again.")
                return
            st.session_state.student_id = student_id
            st.session_state.student_name = data["name"]

        for subj, v in subjects.items():
            submit_preference(student_id, subj, v["teacher_id"], v["priority"])

        if ai_ready(model) and not st.session_state.get("welcome_note"):
            with st.spinner("Your AI concierge is writing you a welcome note…"):
                note = generate_welcome_note(model, data.get("name", ""), list(subjects.keys()))
            if note:
                st.session_state["welcome_note"] = note

        st.session_state.editing_prefs = False
        st.balloons()
        st.rerun()


def _signup_dashboard():
    st.markdown(f"### 🎉 You're all set, {st.session_state.student_name}!")

    srow = get_student_by_id(st.session_state.student_id)
    if srow:
        st.info(f"🪪 **Your Student ID:** `{srow[2]}` — save this! You'll need it to "
                "log back in next time (use it in Step 1's 'Already have a Student ID?' box).")

    if st.session_state.get("welcome_note"):
        st.markdown(f'<div class="sh-note">💬 {st.session_state["welcome_note"]}</div>',
                    unsafe_allow_html=True)

    rows = view_student_preferences(st.session_state.student_id)
    if rows:
        st.markdown("#### Your enrolled subjects")
        for subject_name, teacher_name, priority in rows:
            st.write(f"{priority}. **{subject_name}** → {teacher_name or 'No preference'}")
    else:
        st.info("You haven't picked any subjects yet.")

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("➕ Update subjects / teachers"):
        srow = get_student_by_id(st.session_state.student_id)
        if srow:
            st.session_state.signup_data = {
                "name": srow[1], "roll": srow[2], "grade": srow[3] or "", "email": srow[4] or "",
                "subjects": {
                    subj: {"teacher_id": None, "teacher_name": teacher or "No preference", "priority": pr}
                    for subj, teacher, pr in rows
                },
            }
        st.session_state.editing_prefs = True
        st.session_state.signup_step = 2
        st.rerun()
    if c2.button("Switch student"):
        for k in ("student_id", "student_name", "signup_step", "signup_data",
                  "editing_prefs", "welcome_note"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    with st.expander("⚠️ Delete my profile"):
        st.warning("This permanently deletes your account, along with all your subject and "
                  "teacher preferences. This cannot be undone.")
        confirm = st.checkbox("Yes, I understand this cannot be undone",
                             key="confirm_delete_student")
        if st.button("🗑️ Delete my profile", type="primary", disabled=not confirm,
                    key="delete_student_btn"):
            delete_student(st.session_state.student_id)
            for k in ("student_id", "student_name", "signup_step", "signup_data",
                     "editing_prefs", "welcome_note", "confirm_delete_student"):
                st.session_state.pop(k, None)
            st.success("Your profile has been deleted.")
            st.rerun()


def student_signup_wizard(model):
    """Entry point for the sign-up / preferences tab."""
    _inject_hero_css()
    _render_hero()
    _init_signup_state()

    if st.session_state.get("student_id") and not st.session_state.editing_prefs:
        _signup_dashboard()
        return

    step = st.session_state.signup_step
    _render_steps(step, _STEP_LABELS)
    st.write("")

    if step == 1:
        _step_details()
    elif step == 2:
        _step_subjects()
    elif step == 3:
        _step_confirm(model)


# --------------------------------------------------------------------------- #
def student_view(model):
    # Lectures are locked behind sign-up: a student must have an account and
    # have gone through "Sign up & preferences" before the lecture library
    # becomes visible.
    if not st.session_state.get("student_id"):
        st.info("👋 **Welcome!** Please sign up below to unlock your lectures.")
        student_signup_wizard(model)
        return

    student_id = st.session_state.student_id

    tab_lectures, tab_papers, tab_signup = st.tabs(
        ["🎬 Lectures", "📝 Check my paper", "🚀 Sign up & preferences"]
    )

    with tab_lectures:
        items = load_index()
        if not items:
            st.info("📭 No lectures yet. Ask your teacher to switch to **Teacher** mode and "
                    "upload one!")
        else:
            subjects = sorted({it["subject"] for it in items})
            subject = st.selectbox("📚 Choose a subject", subjects)
            in_subject = [it for it in items if it["subject"] == subject]

            done, total = get_subject_progress(student_id, [it["id"] for it in in_subject])
            if total:
                st.progress(done / total, text=f"{done}/{total} lectures watched in {subject}")

            titles = [it["title"] for it in in_subject]
            picked = st.selectbox("🎬 Choose a lecture", range(len(in_subject)),
                                  format_func=lambda i: titles[i])
            st.divider()
            _play_lecture(in_subject[picked], model, student_id)

    with tab_papers:
        _paper_checker_ui(model, student_id)

    with tab_signup:
        student_signup_wizard(model)


# --------------------------------------------------------------------------- #
# UI — Teacher profile wizard
#
# Mirrors the student sign-up wizard 1:1: details -> subjects taught -> a
# generated "staff ID card" + an AI concierge welcome note. Once a teacher is
# identified, the rest of the teacher dashboard (lecture management, student
# picks) unlocks.
# --------------------------------------------------------------------------- #
_T_STEP_LABELS = ["1 · Your details", "2 · Subjects you teach", "3 · Confirm"]


def _render_teacher_hero():
    st.markdown(
        """<div class="sh-hero">
        <h1>🏫 Manage your classroom</h1>
        <p>Set up your staff profile, add your lectures and papers, and see which students
        pick you as their preferred teacher — all in one place.</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _init_teacher_state():
    st.session_state.setdefault("t_signup_step", 1)
    st.session_state.setdefault("t_signup_data", {})
    st.session_state.setdefault("t_editing_prefs", False)


def _teacher_step_details():
    st.subheader("Step 1 — Tell us about you")
    name = st.text_input("Full name", value=st.session_state.t_signup_data.get("name", ""),
                         placeholder="e.g. Ayesha Khan", key="t_name_field")

    if name.strip() and not st.session_state.t_editing_prefs:
        existing = get_teacher_by_name(name.strip())
        if existing:
            st.info(f"Welcome back, **{existing[1]}**! We'll log you straight in — "
                    "no need to re-enter your details.")
            if st.button("Continue as this teacher →", type="primary"):
                st.session_state.teacher_id = existing[0]
                st.session_state.teacher_name = existing[1]
                st.rerun()
            return

    email = st.text_input("Email address", value=st.session_state.t_signup_data.get("email", ""),
                          placeholder="you@example.com")

    _, c2 = st.columns([1, 1])
    if c2.button("Next: Subjects →", type="primary"):
        if not (name.strip() and email.strip()):
            st.warning("Full name and email are required.")
        elif "@" not in email:
            st.warning("Please enter a valid email address.")
        else:
            st.session_state.t_signup_data.update({"name": name.strip(), "email": email.strip()})
            st.session_state.t_signup_step = 2
            st.rerun()


def _teacher_step_subjects():
    st.subheader("Step 2 — Which subjects do you teach?")
    st.caption("Pick every subject you can teach. Students will see you as a pickable "
              "teacher for each one.")

    prev = st.session_state.t_signup_data.get("subjects", [])
    chosen = st.multiselect("Subjects you teach", SUBJECTS, default=prev)

    c1, c2 = st.columns([1, 1])
    if c1.button("← Back", key="t_back_step2"):
        if st.session_state.t_editing_prefs:
            st.session_state.t_editing_prefs = False
        else:
            st.session_state.t_signup_step = 1
        st.rerun()
    if c2.button("Next: Review →", type="primary", key="t_next_step2"):
        if not chosen:
            st.warning("Pick at least one subject.")
        else:
            st.session_state.t_signup_data["subjects"] = chosen
            st.session_state.t_signup_step = 3
            st.rerun()


def _teacher_step_confirm(model):
    st.subheader("Step 3 — Review & confirm")
    data = st.session_state.t_signup_data
    subjects = data.get("subjects", [])

    initials = "".join(p[0].upper() for p in (data.get("name") or "?").split()[:2]) or "?"
    badges = "".join(f'<span class="sh-subject-badge">{s}</span>' for s in subjects)
    st.markdown(f"""
    <div class="sh-id-card">
      <div class="sh-id-avatar">{initials}</div>
      <div class="sh-id-name">{data.get('name', '')}</div>
      <div class="sh-id-meta">Staff · {data.get('email', '')}</div>
      <div>{badges}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1, 1])
    if c1.button("← Back", key="t_back_step3"):
        st.session_state.t_signup_step = 2
        st.rerun()
    if c2.button("✅ Confirm & save profile", type="primary", key="t_confirm"):
        if st.session_state.get("teacher_id") and st.session_state.t_editing_prefs:
            teacher_id = st.session_state.teacher_id
            set_teacher_subjects(teacher_id, subjects)
            if data.get("email"):
                register_teacher_for_subjects(data["name"], data["email"], [])
        else:
            teacher_id = register_teacher_for_subjects(data["name"], data["email"], subjects)
            st.session_state.teacher_id = teacher_id
            st.session_state.teacher_name = data["name"]

        if ai_ready(model) and not st.session_state.get("teacher_welcome_note"):
            with st.spinner("Your AI concierge is writing you a welcome note…"):
                note = generate_teacher_welcome_note(model, data.get("name", ""), subjects)
            if note:
                st.session_state["teacher_welcome_note"] = note

        st.session_state.t_editing_prefs = False
        st.balloons()
        st.rerun()


def _teacher_profile_dashboard():
    st.markdown(f"### 🎉 Welcome, {st.session_state.teacher_name}!")

    if st.session_state.get("teacher_welcome_note"):
        st.markdown(f'<div class="sh-note">💬 {st.session_state["teacher_welcome_note"]}</div>',
                    unsafe_allow_html=True)

    subjects = get_subjects_for_teacher(st.session_state.teacher_id)
    if subjects:
        st.markdown("#### Subjects you teach")
        for s in subjects:
            st.write(f"- {s}")
    else:
        st.info("You haven't added any subjects yet.")

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("➕ Update my subjects"):
        trow = get_teacher_by_id(st.session_state.teacher_id)
        st.session_state.t_signup_data = {
            "name": trow[1] if trow else st.session_state.teacher_name,
            "email": trow[2] if trow else "",
            "subjects": subjects,
        }
        st.session_state.t_editing_prefs = True
        st.session_state.t_signup_step = 2
        st.rerun()
    if c2.button("Switch teacher"):
        for k in ("teacher_id", "teacher_name", "t_signup_step", "t_signup_data",
                  "t_editing_prefs", "teacher_welcome_note"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    with st.expander("⚠️ Delete my profile"):
        st.warning("This permanently deletes your teacher account and your subject links. "
                  "Lectures and papers you've already added are kept (not deleted) so "
                  "students don't lose access to them. This cannot be undone.")
        confirm = st.checkbox("Yes, I understand this cannot be undone",
                             key="confirm_delete_teacher")
        if st.button("🗑️ Delete my profile", type="primary", disabled=not confirm,
                    key="delete_teacher_btn"):
            delete_teacher(st.session_state.teacher_id)
            for k in ("teacher_id", "teacher_name", "t_signup_step", "t_signup_data",
                     "t_editing_prefs", "teacher_welcome_note", "confirm_delete_teacher"):
                st.session_state.pop(k, None)
            st.success("Your profile has been deleted.")
            st.rerun()


def teacher_profile_wizard(model):
    """Entry point for the teacher profile tab — mirrors student_signup_wizard."""
    _inject_hero_css()
    _render_teacher_hero()
    _init_teacher_state()

    if st.session_state.get("teacher_id") and not st.session_state.t_editing_prefs:
        _teacher_profile_dashboard()
        return

    step = st.session_state.t_signup_step
    _render_steps(step, _T_STEP_LABELS)
    st.write("")

    if step == 1:
        _teacher_step_details()
    elif step == 2:
        _teacher_step_subjects()
    elif step == 3:
        _teacher_step_confirm(model)


# --------------------------------------------------------------------------- #
# UI — Teacher: manage lectures (upload / edit / delete) + student picks
# --------------------------------------------------------------------------- #
def _teacher_manage_lectures(teacher_id):
    st.subheader("📚 Manage your lectures")

    with st.expander("➕ Add a new lecture", expanded=False):
        with st.form("upload", clear_on_submit=True):
            title = st.text_input("Lecture title", placeholder="e.g. Photosynthesis — Part 1")
            subject = st.selectbox("Subject", SUBJECTS)
            description = st.text_input("One-line description", placeholder="What is this lesson about?")
            notes = st.text_area("Lecture notes (the AI tutor & quiz use these)", height=180,
                                 placeholder="Paste or write the key notes for this lecture…")
            video = st.file_uploader("Video lecture", type=VIDEO_TYPES)
            submitted = st.form_submit_button("⬆️ Upload lecture", type="primary")
            if submitted:
                if not title.strip() or video is None:
                    st.warning("Please give a title and choose a video file.")
                else:
                    with st.spinner("Saving…"):
                        add_lecture(title, subject, description, notes, video, teacher_id=teacher_id)
                    st.success(f"Uploaded “{title.strip()}” to {subject}.")
                    st.rerun()

    items = [it for it in load_index() if it.get("teacher_id") == teacher_id]
    st.markdown(f"#### Your lectures ({len(items)})")

    if not items:
        st.info("You haven't uploaded any lectures yet — add your first one above!")
        return

    for it in reversed(items):
        editing_key = f"editing_{it['id']}"
        st.markdown('<div class="sh-lecture-card">', unsafe_allow_html=True)
        header_c1, header_c2, header_c3 = st.columns([5, 1, 1])
        header_c1.markdown(f"**{it['title']}** · {it['subject']}  \n"
                           f"<span style='opacity:.7'>{it.get('description','')} · "
                           f"uploaded {it.get('uploaded_at','')}</span>", unsafe_allow_html=True)
        if header_c2.button("✏️ Edit", key=f"editbtn_{it['id']}"):
            st.session_state[editing_key] = not st.session_state.get(editing_key, False)
        if header_c3.button("🗑️ Delete", key=f"del_{it['id']}"):
            delete_lecture(it["id"])
            st.rerun()

        if st.session_state.get(editing_key):
            with st.form(f"edit_form_{it['id']}"):
                e_title = st.text_input("Title", value=it["title"], key=f"e_title_{it['id']}")
                e_subject = st.selectbox("Subject", SUBJECTS,
                                         index=SUBJECTS.index(it["subject"]) if it["subject"] in SUBJECTS else 0,
                                         key=f"e_subject_{it['id']}")
                e_description = st.text_input("Description", value=it.get("description", ""),
                                              key=f"e_desc_{it['id']}")
                e_notes = st.text_area("Notes", value=it.get("notes", ""), height=160,
                                       key=f"e_notes_{it['id']}")
                save = st.form_submit_button("💾 Save changes", type="primary",
                                             key=f"save_{it['id']}")
                if save:
                    edit_lecture(it["id"], e_title, e_subject, e_description, e_notes)
                    st.session_state[editing_key] = False
                    st.success("Lecture updated.")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def _teacher_manage_papers(teacher_id):
    st.subheader("📝 Manage papers")
    st.caption("Add a paper here (the questions, and — optionally — a marking scheme) so "
              "students can upload their solved version and get it checked instantly by AI.")

    with st.expander("➕ Add a new paper", expanded=False):
        with st.form("upload_paper", clear_on_submit=True):
            title = st.text_input("Paper title", placeholder="e.g. Chapter 4 Practice Test")
            subject = st.selectbox("Subject", SUBJECTS, key="paper_subject")
            description = st.text_input("One-line description", placeholder="What does this paper cover?")
            marking_notes = st.text_area(
                "Marking scheme / notes (optional but recommended)", height=140,
                placeholder="Paste the answer key or grading criteria here — the more detail, "
                            "the more accurate the AI's marking will be. Leave blank to let the "
                            "AI mark using general O-Level standards.")
            paper_file = st.file_uploader("Paper file (questions)", type=PAPER_TYPES,
                                          key="paper_file_uploader")
            submitted = st.form_submit_button("⬆️ Add paper", type="primary")
            if submitted:
                if not title.strip() or paper_file is None:
                    st.warning("Please give a title and upload the paper file.")
                elif paper_file.name.split(".")[-1].lower() == "pdf" and not PDF_SUPPORT:
                    st.error("PDF support isn't installed in this environment "
                             "(`pip install pymupdf`). Please upload a JPG/PNG instead.")
                else:
                    with st.spinner("Saving…"):
                        add_paper(title, subject, description, marking_notes, paper_file, teacher_id)
                    st.success(f"Added “{title.strip()}” to {subject}.")
                    st.rerun()

    items = get_papers_for_teacher(teacher_id)
    st.markdown(f"#### Your papers ({len(items)})")

    if not items:
        st.info("You haven't added any papers yet — add your first one above!")
        return

    for it in reversed(items):
        editing_key = f"editing_paper_{it['id']}"
        st.markdown('<div class="sh-lecture-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([5, 1, 1])
        c1.markdown(f"**{it['title']}** · {it['subject']}  \n"
                   f"<span style='opacity:.7'>{it.get('description','')} · "
                   f"uploaded {it.get('uploaded_at','')} · "
                   f"{'has marking scheme' if it.get('marking_notes') else 'no marking scheme'}</span>",
                   unsafe_allow_html=True)
        if c2.button("✏️ Edit", key=f"editpaperbtn_{it['id']}"):
            st.session_state[editing_key] = not st.session_state.get(editing_key, False)
        if c3.button("🗑️ Delete", key=f"delpaper_{it['id']}"):
            delete_paper(it["id"])
            st.rerun()

        if st.session_state.get(editing_key):
            with st.form(f"edit_paper_form_{it['id']}"):
                e_title = st.text_input("Title", value=it["title"], key=f"ep_title_{it['id']}")
                e_subject = st.selectbox("Subject", SUBJECTS,
                                         index=SUBJECTS.index(it["subject"]) if it["subject"] in SUBJECTS else 0,
                                         key=f"ep_subject_{it['id']}")
                e_description = st.text_input("Description", value=it.get("description", ""),
                                              key=f"ep_desc_{it['id']}")
                e_marking = st.text_area("Marking scheme / notes", value=it.get("marking_notes", ""),
                                         height=140, key=f"ep_marking_{it['id']}")
                save = st.form_submit_button("💾 Save changes", type="primary",
                                             key=f"epsave_{it['id']}")
                if save:
                    edit_paper(it["id"], e_title, e_subject, e_description, e_marking)
                    st.session_state[editing_key] = False
                    st.success("Paper updated.")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def _teacher_student_picks(teacher_id):
    st.subheader("🎓 Students who picked you")
    rows = get_preferences_for_teacher(teacher_id)
    if not rows:
        st.info("No students have picked you yet.")
        return
    for full_name, roll, subject_name, priority in rows:
        st.write(f"**{full_name}** ({roll}) — {subject_name}, priority {priority}")


def _teacher_stats_bar(teacher_id):
    _inject_hero_css()
    subjects_count = len(get_subjects_for_teacher(teacher_id))
    lectures_count = len([it for it in load_index() if it.get("teacher_id") == teacher_id])
    papers_count = len(get_papers_for_teacher(teacher_id))
    students_count = len({row[1] for row in get_preferences_for_teacher(teacher_id)})  # distinct roll numbers

    stats = [
        ("Subjects you teach", subjects_count, "📚"),
        ("Lectures added", lectures_count, "🎬"),
        ("Papers added", papers_count, "📝"),
        ("Students who picked you", students_count, "🎓"),
    ]
    cards = "".join(
        f'<div class="ol-stat-card"><span class="ol-stat-icon">{icon}</span>'
        f'<div class="ol-stat-label">{label}</div>'
        f'<div class="ol-stat-value">{value}</div></div>'
        for label, value, icon in stats
    )
    st.markdown(f'<div class="ol-stat-grid">{cards}</div>', unsafe_allow_html=True)


def teacher_view(model):
    # Mirrors student_view: a teacher must have a profile set up before the
    # lecture-management dashboard becomes visible.
    if not st.session_state.get("teacher_id"):
        st.info("👋 **Welcome!** Please set up your teacher profile below to start managing lectures.")
        teacher_profile_wizard(model)
        return

    _teacher_stats_bar(st.session_state.teacher_id)

    tab_manage, tab_papers, tab_picks, tab_profile = st.tabs(
        ["📚 Manage lectures", "📝 Manage papers", "🎓 Student picks", "🏫 My profile"]
    )

    with tab_manage:
        _teacher_manage_lectures(st.session_state.teacher_id)

    with tab_papers:
        _teacher_manage_papers(st.session_state.teacher_id)

    with tab_picks:
        _teacher_student_picks(st.session_state.teacher_id)

    with tab_profile:
        teacher_profile_wizard(model)


# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Sharks Academy · Study Hub", page_icon="🦈", layout="wide")

    create_tables()
    seed_subjects()

    _render_org_header()

    st.sidebar.title("🦈 Sharks Academy")
    _render_sidebar_infographic()
    role = _render_role_selector()
    model = DEFAULT_MODEL
    if not ai_ready(model):
        st.sidebar.warning("AI features are off (no key set) — video + notes still work.")
    st.sidebar.caption(f"{len(load_index())} lecture(s) available.")

    if role == "Teacher":
        st.title("👩‍🏫 Teacher dashboard")
        teacher_view(model)
    else:
        st.title("🎬 Study time!")
        st.caption("Pick a subject, watch the lecture, and study with your AI tutor.")
        student_view(model)


if __name__ == "__main__":
    main()
