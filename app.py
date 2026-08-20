import os
import re
import json
import io
import spacy
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load SpaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    raise RuntimeError("SpaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm")

app = FastAPI(title="Candidate Matching & Resume Parsing API")

# Enable CORS so your local HTML frontend can communicate seamlessly with the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SECTIONS = ["CONTACT", "EDUCATION", "EXPERIENCE", "SKILLS", "CERTIFICATIONS", "PROJECTS", "SUMMARY"]

# ==============================================================================
# 1. TEXT EXTRACTION
# ==============================================================================
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    raw_text = ""
    try:
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"
        elif ext in [".docx", ".doc"]:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            for p in doc.paragraphs:
                if p.text:
                    raw_text += p.text + "\n"
        elif ext == ".txt":
            raw_text = file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file '{filename}': {str(e)}")

    return raw_text.strip()

# ==============================================================================
# 2. FLEXIBLE SECTION SEGMENTATION
# ==============================================================================
def segment_resume(raw_text: str):
    lines = raw_text.splitlines()
    segmented = {sec: [] for sec in ALLOWED_SECTIONS}
    current_sec = "SUMMARY"

    section_keywords = {
        "EDUCATION": ["education", "academic", "qualification", "degree"],
        "EXPERIENCE": ["experience", "employment", "work history", "work experience", "career"],
        "SKILLS": ["skills", "technical skills", "technologies", "competencies", "tools"],
        "CERTIFICATIONS": ["certification", "certifications", "licenses", "courses"],
        "PROJECTS": ["projects", "key projects", "academic projects"],
        "CONTACT": ["contact", "personal info"]
    }

    for line in lines:
        clean = line.strip().lower()
        if not clean:
            continue
        
        # Identify section header lines (short lines containing keywords)
        is_header = False
        if len(clean) < 40:
            for sec, keywords in section_keywords.items():
                if any(kw in clean for kw in keywords):
                    current_sec = sec
                    is_header = True
                    break
        
        if not is_header:
            segmented[current_sec].append(line.strip())

    return {sec: "\n".join(lines_list) for sec, lines_list in segmented.items()}

# ==============================================================================
# 3. GROUNDED PROFILE FIELD PARSER
# ==============================================================================
def parse_profile_fields(segmented, raw_text):
    doc = nlp(raw_text[:2000]) # Process header and context using SpaCy NER
    fields = []

    def add_field(f_id, category, status, val, evidence, source):
        fields.append({
            "field_id": f_id,
            "category": category,
            "status": status,
            "value": val if status == "FOUND" else "Not Found",
            "evidence": evidence if status == "FOUND" else "No matching evidence found.",
            "source_section": source
        })

    # 1. Email
    email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', raw_text)
    add_field("EMAIL-1", "EMAIL", "FOUND" if email else "NOT_FOUND",
              email.group(0) if email else None, email.group(0) if email else None, "CONTACT")

    # 2. Phone
    phone = re.search(r'(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', raw_text)
    add_field("PHONE-1", "PHONE", "FOUND" if phone else "NOT_FOUND",
              phone.group(0) if phone else None, phone.group(0) if phone else None, "CONTACT")

    # 3. LinkedIn URL
    linkedin = re.search(r'(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+', raw_text, re.I)
    add_field("LINKEDIN-1", "LINKEDIN_URL", "FOUND" if linkedin else "NOT_FOUND",
              linkedin.group(0) if linkedin else None, linkedin.group(0) if linkedin else None, "CONTACT")

    # 4. Candidate Name (SpaCy Named Entity Recognition)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    name_val = names[0] if names else None
    add_field("NAME-1", "FULL_NAME", "FOUND" if name_val else "AMBIGUOUS",
              name_val, name_val if name_val else "Top section line analyzed", "HEADER")

    # 5. Highest Degree
    edu_text = segmented.get("EDUCATION", "") + "\n" + raw_text
    degree = re.search(r'(?i)\b(Ph\.?D|Doctor|Master|M\.S|M\.Sc|Bachelor|B\.S|B\.Sc|B\.E|B\.Tech|Diploma|Associate)\b', edu_text)
    add_field("DEGREE-1", "HIGHEST_DEGREE", "FOUND" if degree else "NOT_FOUND",
              degree.group(0) if degree else None, degree.group(0) if degree else None, "EDUCATION")

    # 6. Job Title / Role
    exp_text = segmented.get("EXPERIENCE", "")
    exp_lines = [l for l in exp_text.splitlines() if l.strip()]
    job_val = exp_lines[0] if exp_lines else None
    add_field("JOB-1", "MOST_RECENT_JOB", "FOUND" if job_val else "NOT_FOUND",
              job_val, job_val, "EXPERIENCE")

    # 7. Location (SpaCy Geopolitical Entity)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    loc_val = locations[0] if locations else None
    add_field("LOC-1", "LOCATION", "FOUND" if loc_val else "NOT_FOUND",
              loc_val, loc_val, "CONTACT")

    # 8. Skills List
    skills_text = segmented.get("SKILLS", "").strip()
    if skills_text:
        skills_clean = ", ".join([s.strip() for s in re.split(r'[\n,•|;]', skills_text) if s.strip()])
        add_field("SKILLS-LIST", "SKILLS_LIST", "FOUND", skills_clean[:300], skills_text[:200].replace('\n', ' '), "SKILLS")
    else:
        add_field("SKILLS-LIST", "SKILLS_LIST", "NOT_FOUND", None, None, "SKILLS")

    # 9. Certifications
    cert_text = segmented.get("CERTIFICATIONS", "").strip()
    add_field("CERTS-1", "CERTIFICATIONS_LIST", "FOUND" if cert_text else "NOT_FOUND",
              cert_text.replace('\n', '; ')[:150] if cert_text else None,
              cert_text[:150].replace('\n', ' ') if cert_text else None, "CERTIFICATIONS")

    # 10. Key Projects
    proj_text = segmented.get("PROJECTS", "").strip()
    add_field("PROJ-1", "PROJECTS_LIST", "FOUND" if proj_text else "NOT_FOUND",
              proj_text.splitlines()[0] if proj_text else None,
              proj_text[:150].replace('\n', ' ') if proj_text else None, "PROJECTS")

    return fields

# ==============================================================================
# 4. CANDIDATE FIT ASSESSMENT ENGINE
# ==============================================================================
def evaluate_candidate_fit(fields, jd_text):
    fit_items = []
    jd_lower = jd_text.lower()

    # Skill Matching
    skills_field = next((f for f in fields if f["field_id"] == "SKILLS-LIST"), None)
    if skills_field and skills_field["status"] == "FOUND":
        candidate_skills = [s.strip() for s in skills_field["value"].split(",") if s.strip()]
        matched = [s for s in candidate_skills if s.lower() in jd_lower]
        
        if matched:
            fit_items.append({
                "requirement": "Required Technical Skills",
                "match_status": "MATCHED",
                "explanation": f"Matched skills found in JD: {', '.join(set(matched))}.",
                "evidence_ref": "SKILLS-LIST",
                "confidence": "high"
            })
        else:
            fit_items.append({
                "requirement": "Required Technical Skills",
                "match_status": "MISSING",
                "explanation": "No candidate skills directly matched terms in the job description.",
                "evidence_ref": "SKILLS-LIST",
                "confidence": "medium"
            })
    else:
        fit_items.append({
            "requirement": "Required Technical Skills",
            "match_status": "MISSING",
            "explanation": "No skills section extracted from resume.",
            "evidence_ref": "SKILLS-LIST",
            "confidence": "high"
        })

    # Degree Qualification Check
    degree_field = next((f for f in fields if f["field_id"] == "DEGREE-1"), None)
    if degree_field and degree_field["status"] == "FOUND":
        fit_items.append({
            "requirement": "Academic Qualification",
            "match_status": "MATCHED",
            "explanation": f"Verified degree extracted: {degree_field['value']}.",
            "evidence_ref": "DEGREE-1",
            "confidence": "high"
        })
    else:
        fit_items.append({
            "requirement": "Academic Qualification",
            "match_status": "MISSING",
            "explanation": "No recognized degree found in Education section.",
            "evidence_ref": "DEGREE-1",
            "confidence": "high"
        })

    return fit_items

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================
@app.get("/")
def health_check():
    return {"status": "online", "message": "Resume Parser API is running."}

@app.post("/api/parse")
async def parse_candidate_resume(file: UploadFile = File(...), jd_text: str = Form(...)):
    if not file or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Missing resume file or job description text.")

    file_bytes = await file.read()
    raw_text = extract_text_from_file(file_bytes, file.filename)
    
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not read text from file. File might be empty or scanned images.")

    segmented = segment_resume(raw_text)
    profile_fields = parse_profile_fields(segmented, raw_text)
    fit_report = evaluate_candidate_fit(profile_fields, jd_text)

    output = {
        "profile": profile_fields,
        "fit_report": fit_report
    }

    # Export profile.json locally for deliverable submission
    with open("profile.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output