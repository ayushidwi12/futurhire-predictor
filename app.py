from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import fitz  # PyMuPDF
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import random
import json
from datetime import date
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# File to store user credentials
USERS_FILE = 'users.json'

# Load model and preprocessing tools
model = joblib.load("model/hiring_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/feature_scaler.pkl")
college_tier_encoder = joblib.load("model/college_tier_encoder.pkl")

# Skills to extract from text
SKILL_KEYWORDS = [
    'python', 'java', 'go', 'javascript', 'ruby', 'php',
    'sql', 'mysql', 'postgresql', 'oracle', 'mongodb', 'cassandra',
    'aws', 'azure', 'google cloud', 'django', 'flask', 'node.js',
    'ruby on rails', 'restful apis', 'web services',
    'problem-solving', 'analytical thinking', 'attention to detail',
    'collaboration', 'teamwork', 'communication', 'adaptability',
    'time management', 'logical thinking', 'continuous learning',
    'leadership', 'mentoring'
]

# ------------------------- Utility Functions -------------------------

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

def extract_keywords(text, keywords):
    return list({kw for kw in keywords if kw in text})

def extract_certificate_names(text):
    matches = re.findall(r"(certificate.*|certified.*)", text, re.IGNORECASE)
    return list(set([m.strip() for m in matches]))

def save_user(username, password):
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    users[username] = password
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def check_user(username, password):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        return users.get(username) == password
    return False

# ------------------------- Routes -------------------------

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/aptitude_test')
def aptitude_test():
    aptitude_questions = [
        {
            "question": "If the average of 5 consecutive odd numbers is 45, what is the largest number?",
            "options": ["47", "49", "51", "53"],
            "answer": 2
        },
        {
            "question": "Train 120m long running at 60 km/h passes a platform in 30 seconds. Find platform length.",
            "options": ["380m", "300m", "280m", "360m"],
            "answer": 0
        },
        {
            "question": "If 2x + 3y = 12 and x - y = 1, find x.",
            "options": ["3", "2", "4", "5"],
            "answer": 2
        },
        {
            "question": "What is the compound interest on ₹5000 for 2 years at 10% p.a. compounded annually?",
            "options": ["₹1000", "₹1050", "₹1100", "₹1155"],
            "answer": 3
        },
        {
            "question": "If a number is divided by 13, the remainder is 5. What will be the remainder if square is divided by 13?",
            "options": ["12", "11", "10", "25"],
            "answer": 1
        },
        {
            "question": "Which number does not belong? 16, 25, 36, 48, 64",
            "options": ["25", "36", "48", "64"],
            "answer": 2
        },
        {
            "question": "A can complete a job in 10 days, B in 15 days. Together they work 5 days. Work left?",
            "options": ["1/4", "1/3", "1/2", "2/3"],
            "answer": 0
        },
        {
            "question": "Simplify: (81)^0.5 × (9)^-1",
            "options": ["1", "3", "2", "0.5"],
            "answer": 0
        },
        {
            "question": "A clock gains 2 minutes every hour. In 24 hours it will gain:",
            "options": ["24 min", "48 min", "50 min", "60 min"],
            "answer": 1
        },
        {
            "question": "If x^2 - 5x + 6 = 0, what are the roots?",
            "options": ["2, 3", "1, 6", "3, 5", "2, 4"],
            "answer": 0
        },
        {
            "question": "Pointing to a woman, Ravi says, ‘She is the only daughter of my father’s only daughter.’ Who is she?",
            "options": ["Sister", "Niece", "Cousin", "Daughter"],
            "answer": 3
        },
        {
            "question": "Find the missing number in the series: 2, 6, 12, 20, ?",
            "options": ["28", "30", "32", "34"],
            "answer": 0
        },
        {
            "question": "If 40% of a number is 80, what is the number?",
            "options": ["100", "150", "200", "180"],
            "answer": 2
        },
        {
            "question": "Raju can complete a work in 8 days. Ramesh in 12 days. Together in?",
            "options": ["4.8", "5.2", "5", "6"],
            "answer": 0
        },
        {
            "question": "What is the angle between the hands of a clock at 3:15?",
            "options": ["7.5°", "15°", "22.5°", "30°"],
            "answer": 2
        },
        {
            "question": "Find the odd one out: 7, 11, 13, 17, 21, 23",
            "options": ["11", "13", "17", "21"],
            "answer": 3
        },
        {
            "question": "Statement: All apples are fruits. Some fruits are sweet. Conclusion: Some apples are sweet.",
            "options": ["True", "False", "Cannot be determined", "All are true"],
            "answer": 2
        },
        {
            "question": "Which word cannot be formed from ‘EDUCATION’?",
            "options": ["CUTE", "DATE", "DINE", "CAUTION"],
            "answer": 2
        },
        {
            "question": "Which number comes next: 2, 3, 5, 7, 11, 13, ?",
            "options": ["15", "17", "16", "19"],
            "answer": 1
        },
        {
            "question": "Which letter is 5th to the right of the 12th letter from left in A-Z?",
            "options": ["R", "Q", "N", "O"],
            "answer": 3
        },
        {
            "question": "If each side of square is increased by 10%, area increases by?",
            "options": ["10%", "20%", "21%", "22%"],
            "answer": 2
        },
        {
            "question": "Select the odd one: Circle, Triangle, Square, Cube",
            "options": ["Circle", "Triangle", "Square", "Cube"],
            "answer": 3
        },
        {
            "question": "A bag contains 3 red, 5 blue balls. Probability of picking red?",
            "options": ["3/8", "3/5", "5/8", "2/5"],
            "answer": 0
        },
        {
            "question": "Find the next number: 1, 4, 9, 16, ?",
            "options": ["25", "24", "36", "30"],
            "answer": 0
        },
        {
            "question": "Sum of interior angles of a hexagon is:",
            "options": ["540°", "720°", "900°", "1080°"],
            "answer": 1
        },
        {
            "question": "If x = 5 and y = 2, then (x^2 + y^2) = ?",
            "options": ["29", "25", "21", "30"],
            "answer": 0
        },
        {
            "question": "If 1 inch = 2.54 cm, then 10 inches = ? cm",
            "options": ["25.4", "24.5", "26.4", "22.5"],
            "answer": 0
        },
        {
            "question": "Which number will come next: 1, 2, 6, 24, ?",
            "options": ["60", "100", "120", "150"],
            "answer": 2
        },
        {
            "question": "If A = 1, B = 2, ..., Z = 26. Value of CAT = ?",
            "options": ["24", "48", "42", "45"],
            "answer": 2
        },
        {
            "question": "Which figure has the most sides?",
            "options": ["Hexagon", "Heptagon", "Nonagon", "Octagon"],
            "answer": 2
        }
    ]
    random.shuffle(aptitude_questions)
    selected = aptitude_questions[:10]
    return render_template('aptitude_test.html', questions=selected)

# New: Utility for parsing ATS (placed after imports)
def parse_ats_resume(file):
    try:
        text = extract_text_from_pdf(file)
        skills = extract_keywords(text, SKILL_KEYWORDS)
        name = re.search(r"name[:\-]?\s*(\w+\s\w+)", text, re.IGNORECASE)
        email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        edu_match = re.search(r"(b\.?e\.?|bachelor.*engineering)", text, re.IGNORECASE)

        return {
            'name': name.group(1) if name else "Not Found",
            'email': email.group(0) if email else "Not Found",
            'skills': skills,
            'education': edu_match.group(0) if edu_match else "Not Found",
            'score': min(100, len(skills) * 5),  # basic scoring logic
            'formatting': 'Good'  # Placeholder for now
        }
    except Exception as e:
        return {
            'name': "Error",
            'email': "Error",
            'skills': [],
            'education': "Error",
            'score': 0,
            'formatting': f"Parsing Error: {e}"
        }

@app.route('/check_ats', methods=['GET', 'POST'])
def check_ats():
    ats_result = None
    if request.method == 'POST':
        resume_file = request.files.get('resume_file')
        if resume_file and resume_file.filename.endswith('.pdf'):
            ats_result = parse_ats_resume(resume_file)
        else:
            ats_result = {
                'name': 'Error',
                'email': 'Invalid or missing PDF file',
                'skills': [],
                'education': 'N/A',
                'score': 0,
                'formatting': 'Invalid file'
            }
    return render_template('check_ats.html', ats_result=ats_result)
@app.route('/download_resume')
def download_resume():
    return send_from_directory('generated_resumes', 'resume.pdf', as_attachment=True)

@app.route('/create_ats', methods=['GET', 'POST'])
def create_ats():
    if request.method == 'POST':
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        summary = request.form.get('summary', '')
        skills = request.form.get('skills', '')
        education = request.form.get('education', '')
        experience = request.form.get('experience', '')
        certifications = request.form.get('certifications', '')

        return render_template('resume_preview.html',
                               name=name,
                               email=email,
                               phone=phone,
                               summary=summary,
                               skills=skills,
                               education=education,
                               experience=experience,
                               certifications=certifications)
    else:
        return render_template('create_ats.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('guest') == 'yes':
            session['user'] = 'Guest'
            return redirect(url_for('home'))

        username = request.form.get('username')
        password = request.form.get('password')

        # Admin hardcoded access
        if username == 'admin' and password == 'password123':
            session['user'] = username
            return redirect(url_for('home'))

        # Check user from file
        if check_user(username, password):
            session['user'] = username
            return redirect(url_for('home'))

        return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")

        save_user(username, password)
        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/preview_resume', methods=['POST'])
def preview_resume():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    summary = request.form.get('summary')
    skills = request.form.get('skills')  # This must be passed
    education = request.form.get('education')
    experience = request.form.get('experience', '')
    certifications = request.form.get('certifications', '')

    return render_template('resume_preview.html',
                           name=request.form.get('name'),
                           email=request.form.get('email'),
                           phone=request.form.get('phone'),
                           summary=request.form.get('summary'),
                           skills=request.form.get('skills', ''),
                           education=request.form.get('education'),
                           experience=request.form.get('experience', ''),
                           certifications=request.form.get('certifications', ''))

@app.route('/companies')
def companies():
    return render_template('companies.html')

@app.route('/more_info')
def more_info():
    return render_template('more_info.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Load walk-in events
    walkins = []
    try:
        with open('walkins.json') as f:
            all_events = json.load(f)
            today = date.today().isoformat()
            walkins = [e for e in all_events if e['date'] >= today]
    except Exception as e:
        walkins = []

    return render_template('home.html', user=session['user'], walkins=walkins)


@app.route('/parse_certificates', methods=['POST'])
def parse_certificates():
    uploaded_files = request.files.getlist('cert_file')
    certificate_names = []
    for file in uploaded_files:
        if file.filename.endswith('.pdf'):  # Changed from allowed_file() to direct check
            text = extract_text_from_pdf(file)
            names = extract_certificate_names(text)
            certificate_names.extend(names)
    return render_template('input.html', parsed_skills=session.get('parsed_skills', []), certificate_names=certificate_names)

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if 'user' not in session:
        return redirect(url_for('login'))

    parsed_skills = []

    if request.method == 'POST':
        if 'resume_file' in request.files:
            resume_file = request.files['resume_file']
            resume_text = extract_text_from_pdf(resume_file)
            parsed_skills = extract_keywords(resume_text, SKILL_KEYWORDS)

    return render_template('input.html', user=session['user'], parsed_skills=parsed_skills)


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        resume_text = ""
        cert_text = ""
        be_result_text = ""

        # Resume
        if 'resume_file' in request.files:
            resume_file = request.files['resume_file']
            if resume_file and resume_file.filename != '':
                resume_text = extract_text_from_pdf(resume_file)

        # Certificates
        if 'cert_file' in request.files:
            cert_file = request.files['cert_file']
            if cert_file and cert_file.filename != '':
                cert_text = extract_text_from_pdf(cert_file)

        # BE result
        if 'be_result_file' in request.files:
            be_result_file = request.files['be_result_file']
            if be_result_file and be_result_file.filename != '':
                be_result_text = extract_text_from_pdf(be_result_file)

        # Combine text for TF-IDF
        combined_text = resume_text + " " + cert_text

        # Numeric inputs
        try:
            aptitude_score = float(request.form.get("aptitude_score", 0))
        except ValueError:
            aptitude_score = 0.0

        try:
            cgpa = float(request.form.get("cgpa", 0))
        except ValueError:
            cgpa = 0.0

        # Categorical input
        college_tier = request.form.get("college_tier", "Other")

        # Feature transforms
        X_text = tfidf.transform([combined_text])

        X_num_raw = np.array([[cgpa, aptitude_score]])
        X_num_scaled = scaler.transform(X_num_raw)
        X_num_sparse = csr_matrix(X_num_scaled)

        try:
            X_college_tier = college_tier_encoder.transform([[college_tier]])
        except Exception:
            # Create zero vector with same shape as one-hot encoder output
            X_college_tier = csr_matrix((1, college_tier_encoder.transform([college_tier_encoder.categories_[0]]).shape[1]))

        # Combine all features horizontally
        final_input = hstack([X_text, X_num_sparse, X_college_tier])

        probabilities = model.predict_proba(final_input)[0]

        # Get top 3 companies by probability
        top_indices = probabilities.argsort()[::-1][:3]
        top_companies = label_encoder.inverse_transform(top_indices)
        top_scores = probabilities[top_indices]

        results = list(zip(top_companies, top_scores))

        # Extract skills from combined text (assuming you have this function and skill list)
        all_text = resume_text + " " + cert_text + " " + be_result_text
        extracted_skills = extract_keywords(all_text, SKILL_KEYWORDS)  # You need to define this

        return render_template("result.html",
                               results=results,
                               extracted_skills=extracted_skills,
                               cgpa=cgpa,
                               aptitude_score=aptitude_score,
                               college_tier=college_tier)
    except Exception as e:
        return f"Error during prediction: {str(e)}"



@app.route('/parse_resume_skills', methods=['POST'])
def parse_resume_skills():
    if 'resume_file' in request.files:
        resume_file = request.files['resume_file']
        resume_text = extract_text_from_pdf(resume_file)
        skills = extract_keywords(resume_text, SKILL_KEYWORDS)
        return {'skills': skills}
    return {'skills': []}

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ------------------------- Run App -------------------------

if __name__ == '__main__':
    app.run(debug=True)
