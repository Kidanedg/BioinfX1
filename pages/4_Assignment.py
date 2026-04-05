import streamlit as st
import random
import pandas as pd
import numpy as np
import py3Dmol
from Bio.PDB import PDBParser
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import datetime

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Bio-Molecular Modeler Pro", layout="wide")
st.title("🧬 Bio-Molecular Modeler + Assignment System")

# =========================================================
# DATABASE
# =========================================================
if "students_db" not in st.session_state:
    st.session_state.students_db = {}

if "deadline" not in st.session_state:
    st.session_state.deadline = datetime.datetime.now() + datetime.timedelta(days=1)

# =========================================================
# LOGIN
# =========================================================
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    role = st.selectbox("Role", ["Student", "Instructor"])
    name = st.text_input("Name")
    sid = st.text_input("ID")

    if st.button("Login") and name and sid:
        st.session_state.user = {"name": name, "id": sid, "role": role}

        if sid not in st.session_state.students_db:
            st.session_state.students_db[sid] = {
                "name": name,
                "history": [],
                "score": 0
            }
    st.stop()

user = st.session_state.user

# =========================================================
# INSTRUCTOR PANEL
# =========================================================
if user["role"] == "Instructor":

    st.header("📊 Instructor Dashboard")

    new_deadline = st.datetime_input("Deadline", st.session_state.deadline)
    st.session_state.deadline = new_deadline

    db = st.session_state.students_db

    data = []
    for sid, info in db.items():
        data.append({
            "ID": sid,
            "Name": info["name"],
            "Score": info["score"],
            "Attempts": len(info["history"])
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    st.subheader("🏆 Leaderboard")
    st.dataframe(df.sort_values(by="Score", ascending=False))

    st.stop()

# =========================================================
# FORCE FIELD PARSER
# =========================================================
def parse_force_field(file):
    lines = file.read().decode().splitlines()
    ff = {}
    section = None

    for line in lines:
        line = line.strip()

        if not line or line.startswith(";"):
            continue

        if line.startswith("["):
            section = line.strip("[]").lower()
            ff[section] = []
        else:
            ff[section].append(line.split())

    return ff

# Sidebar upload
st.sidebar.subheader("⚙️ Force Field")
ff_file = st.sidebar.file_uploader("Upload FF (AMBER/CHARMM style)")

if ff_file:
    st.session_state.ff = parse_force_field(ff_file)
    st.sidebar.success("Force Field Loaded")

# =========================================================
# SESSION STATE
# =========================================================
for k in ["coords", "problem", "true"]:
    if k not in st.session_state:
        st.session_state[k] = None

# =========================================================
# REAL ENERGY FUNCTIONS (FF-BASED)
# =========================================================
def bond_energy(coords, ff):
    if not ff or "bonds" not in ff:
        return 0

    E = 0
    for row in ff["bonds"]:
        try:
            i, j = int(row[0]), int(row[1])
            r0 = float(row[-2])
            k = float(row[-1])

            r = np.linalg.norm(coords[i] - coords[j])
            E += k * (r - r0) ** 2
        except:
            pass
    return E


def angle_energy(coords, ff):
    if not ff or "angles" not in ff:
        return 0

    E = 0
    for row in ff["angles"]:
        try:
            i, j, k_idx = map(int, row[:3])
            theta0 = float(row[-2])
            k = float(row[-1])

            v1 = coords[i] - coords[j]
            v2 = coords[k_idx] - coords[j]

            theta = np.degrees(np.arccos(
                np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            ))

            E += k * (theta - theta0) ** 2
        except:
            pass
    return E


def dihedral_energy(coords, ff):
    if not ff or "dihedrals" not in ff:
        return 0

    E = 0
    for row in ff["dihedrals"]:
        try:
            i, j, k, l = map(int, row[:4])
            k_phi = float(row[-2])
            n = float(row[-1])

            p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[l]

            b0 = p1 - p0
            b1 = p2 - p1
            b2 = p3 - p2

            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)

            phi = np.degrees(np.arccos(
                np.dot(n1, n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
            ))

            E += k_phi * (1 + np.cos(np.radians(n * phi)))
        except:
            pass
    return E


def lj_energy(coords, ff):
    if not ff or "lj" not in ff:
        return 0

    epsilon = float(ff["lj"][0][0])
    sigma = float(ff["lj"][0][1])

    E = 0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            r = np.linalg.norm(coords[i] - coords[j])
            if r == 0:
                continue
            E += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

    return E


def total_energy(coords):
    ff = st.session_state.get("ff", None)

    return {
        "Bond": bond_energy(coords, ff),
        "Angle": angle_energy(coords, ff),
        "Dihedral": dihedral_energy(coords, ff),
        "LJ": lj_energy(coords, ff)
    }

# =========================================================
# AI GRADING ASSISTANT
# =========================================================
def ai_feedback(true, answer, problem_type):
    error = abs(true - answer)

    if error < 0.5:
        return "✅ Excellent! Your calculation is very accurate."
    elif error < 5:
        return "👍 Close! Check rounding or constants."
    else:
        if problem_type == "Bond":
            return "⚠️ Check bond formula: E = k(r - r0)^2"
        elif problem_type == "Angle":
            return "⚠️ Check angle units (degrees vs radians)."
        elif problem_type == "Dihedral":
            return "⚠️ Remember torsion uses cosine periodicity."
        else:
            return "⚠️ Check Lennard-Jones equation carefully."

# =========================================================
# PDB UPLOAD
# =========================================================
uploaded = st.file_uploader("Upload PDB", type=["pdb"])

if uploaded:
    pdb = uploaded.read().decode()

    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb, "pdb")
    view.setStyle({"cartoon": {}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=400)

    with open("temp.pdb", "w") as f:
        f.write(pdb)

    parser = PDBParser(QUIET=True)
    s = parser.get_structure("m", "temp.pdb")

    coords = [a.get_coord() for a in s.get_atoms()]
    st.session_state.coords = np.array(coords)

# =========================================================
# ASSIGNMENT
# =========================================================
st.subheader("🎓 Assignment")

problem_type = st.selectbox("Problem", ["Bond", "Angle", "Dihedral", "LJ"])

if st.button("Generate Problem"):

    if problem_type == "Bond":
        r = random.uniform(1,2)
        true = 300*(r-1.5)**2
        q = f"r={r:.3f}"

    elif problem_type == "Angle":
        t = random.uniform(90,130)
        true = 40*(t-109.5)**2
        q = f"θ={t:.2f}"

    elif problem_type == "Dihedral":
        phi = random.uniform(0,180)
        true = 2*(1 + np.cos(np.radians(phi)))
        q = f"φ={phi:.2f}"

    else:
        r = random.uniform(2.5,6)
        true = 4*0.2*((3.5/r)**12-(3.5/r)**6)
        q = f"r={r:.3f}"

    st.session_state.problem = q
    st.session_state.true = true

# Solve
if st.session_state.problem:
    st.info(st.session_state.problem)

    ans = st.number_input("Your Answer")
    tol = st.slider("Tolerance", 0.0, 10.0, 1.0)

    if st.button("Submit"):

        now = datetime.datetime.now()
        deadline = st.session_state.deadline

        error = abs(ans - st.session_state.true)

        score = 100 if error <= tol else max(0, 100 - error)

        late = now > deadline
        if late:
            score *= 0.7

        feedback = ai_feedback(st.session_state.true, ans, problem_type)

        sid = user["id"]
        st.session_state.students_db[sid]["history"].append({
            "time": now,
            "type": problem_type,
            "error": error,
            "score": score,
            "late": late
        })

        st.session_state.students_db[sid]["score"] += score

        st.success(f"Score: {score:.2f}")
        st.info(feedback)

        if late:
            st.warning("Late Submission Penalty Applied")

# =========================================================
# ENERGY ANALYSIS
# =========================================================
if st.session_state.coords is not None:

    st.subheader("⚛️ Energy Decomposition")

    E = total_energy(st.session_state.coords)

    df = pd.DataFrame(list(E.items()), columns=["Component", "Energy"])
    st.dataframe(df)
    st.bar_chart(df.set_index("Component"))

# =========================================================
# STUDENT DASHBOARD
# =========================================================
st.header("📈 Your Performance")

sid = user["id"]
hist = pd.DataFrame(st.session_state.students_db[sid]["history"])

if not hist.empty:
    st.dataframe(hist)
    st.line_chart(hist["score"])

# =========================================================
# EXPORT
# =========================================================
if st.button("Export Report"):

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.drawString(100, 750, f"Student: {user['name']}")
    c.drawString(100, 730, f"Score: {st.session_state.students_db[sid]['score']}")

    c.save()
    buffer.seek(0)

    st.download_button("Download PDF", buffer, "report.pdf")
