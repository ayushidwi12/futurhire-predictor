import fitz
import re

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    known_skills = ["python", "java", "c++", "sql", "html", "css", "machine learning", "deep learning", "flask", "django"]
    text = text.lower()
    return [skill for skill in known_skills if skill in text]

def extract_certificates(text):
    cert_keywords = ["certified", "certification", "course", "coursera", "udemy", "oracle", "google", "aws"]
    text = text.lower()
    return [line for line in text.split('\n') if any(keyword in line for keyword in cert_keywords)]

def extract_internships(text):
    internships = re.findall(r"intern(?:ship)?(?: at)? ([A-Za-z0-9 &]+)", text, re.IGNORECASE)
    return internships

def extract_projects(text):
    project_keywords = ["project", "built", "developed", "created", "designed"]
    return [line for line in text.split('\n') if any(word in line.lower() for word in project_keywords)]
