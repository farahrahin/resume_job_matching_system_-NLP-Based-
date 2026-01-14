import streamlit as st
import pandas as pd
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─── NLTK & Models ───────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume → Job Matcher (Skill Focused)",
    layout="wide",
    page_icon="📄✨"
)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("Resume – Job Matching System")
st.caption("Malaysia-focused • Skill overlap driven matching • JobStreet data")
st.markdown("---")

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data
def load_jobs():
    try:
        # CHANGE THIS PATH according to your file location
        path = "jobstreet_all_job.csv"
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        
        # Rename important columns
        rename_dict = {
            'job_title': 'Job Title',
            'descriptions': 'Job Description',
            'Company': 'Company',
            'location': 'Location',
            'category': 'Category',
            'type': 'Job Type',
            'salary': 'Salary'
        }
        df = df.rename(columns=rename_dict)
        
        required = ['Job Title', 'Job Description']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
            
        df = df.dropna(subset=['Job Description'])
        st.success(f"Loaded {len(df):,} job postings")
        return df
        
    except Exception as e:
        st.error(f"Cannot load data\n{e}")
        st.stop()

jobs = load_jobs()

# ─── SKILL VOCABULARY ────────────────────────────────────────────────────────
@st.cache_data
def build_skill_vocab():
    text = " ".join(jobs["Job Description"].dropna()).lower()
    tokens = [t for t in word_tokenize(text) if t.isalpha() and len(t) >= 3]
    
    count = Counter(tokens)
    
    # Very common non-skill words to filter out
    very_generic = {
        'and','the','for','with','from','this','that','new','all','any',
        'our','you','will','are','about','one','part','based','able','high',
        'level','time','including','ensure','excellent','strong','good','work',
        'working','career','growth','experience','education','degree',
        'professional','perform','performance','quality','support','system'
    }
    
    candidates = [w for w,c in count.most_common(800) if w not in very_generic and c >= 6]
    
    # Add some common multi-word / important skills manually (Malaysia context)
    extra_important = [
        'communication skills', 'team work', 'teamwork', 'problem solving',
        'project management', 'data analysis', 'customer service',
        'microsoft excel', 'microsoft office', 'power bi', 'sql', 'python',
        'java', 'javascript', 'react', 'flutter', 'ui ux', 'graphic design'
    ]
    
    vocab = sorted(list(set(candidates + extra_important)))[:350]
    return vocab

skill_vocab = build_skill_vocab()

st.sidebar.header("Statistics")
st.sidebar.metric("Jobs in database", f"{len(jobs):,}")
st.sidebar.metric("Learned skill terms", len(skill_vocab))
if skill_vocab:
    st.sidebar.caption("Some detected skills:")
    st.sidebar.write(", ".join(skill_vocab[:12]) + " ...")

# ─── FUNCTIONS ───────────────────────────────────────────────────────────────
def extract_text(file):
    text = ""
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            text = file.read().decode("utf-8", errors="ignore")
        return text.strip()
    except:
        return ""

def extract_resume_skills(text):
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for skill in skill_vocab:
        if skill in text_lower:
            found.add(skill)
    return sorted(list(found))

def get_matched_skills(resume_skills, job_desc):
    if not job_desc:
        return []
    text = str(job_desc).lower()
    matched = []
    for s in resume_skills:
        # Basic phrase matching + some common variations
        if s in text:
            matched.append(s)
        elif s == "communication skills" and any(x in text for x in [
            "communication skill", "communication skills", "strong communication",
            "excellent communication", "good communication", "communicate effectively"
        ]):
            matched.append("communication skills")
        elif s == "teamwork" and "team" in text:
            matched.append("teamwork")
        elif s == "problem solving" and any(x in text for x in ["problem solving", "problem-solving"]):
            matched.append("problem solving")
    return sorted(list(set(matched)))

def get_similarity(text1, text2):
    try:
        emb = embedder.encode([text1[:3000], text2[:3000]])  # truncate long texts
        return cosine_similarity([emb[0]], [emb[1]])[0][0]
    except:
        return 0.0

def calculate_score(semantic_sim, skill_overlap_ratio):
    # Give more weight to actual skill matches
    return round(0.45 * semantic_sim + 0.55 * skill_overlap_ratio, 3)

# ─── MAIN INTERFACE ──────────────────────────────────────────────────────────
resume_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

if resume_file:
    with st.spinner("Reading resume..."):
        resume_text = extract_text(resume_file)
    
    if not resume_text.strip():
        st.error("Could not extract any text from the file.")
    else:
        st.success("Resume loaded successfully!")
        
        with st.spinner("Extracting skills from your resume..."):
            resume_skills = extract_resume_skills(resume_text)
        
        if not resume_skills:
            st.warning("No recognizable skills found in your resume.")
        else:
            st.subheader("Skills detected in your resume")
            st.markdown(
                "  •  ".join([f"**{s}**" for s in resume_skills]),
                unsafe_allow_html=True
            )
            st.caption(f"Total: {len(resume_skills)} skills")

        # ── JOB MATCHING ─────────────────────────────────────────────────────
        if resume_skills:
            st.markdown("---")
            st.subheader("Matching Jobs (sorted by skill overlap + semantic similarity)")

            results = []

            with st.spinner("Comparing with jobs..."):
                for _, job in jobs.iterrows():
                    matched = get_matched_skills(resume_skills, job["Job Description"])
                    if len(matched) == 0:
                        continue

                    semantic = get_similarity(resume_text, job["Job Description"])
                    skill_ratio = len(matched) / len(resume_skills)
                    score = calculate_score(semantic, skill_ratio)

                    results.append({
                        "Job Title": job["Job Title"],
                        "Company": job.get("Company", "—"),
                        "Location": job.get("Location", "—"),
                        "Category": job.get("Category", "—"),
                        "Salary": job.get("Salary", "—"),
                        "Score": score,
                        "Score %": round(score * 100, 1),
                        "Matched Skills": matched,
                        "Count": len(matched),
                        "Description Preview": str(job["Job Description"])[:320] + "..."
                    })

            if not results:
                st.info("No jobs with matching skills found.")
            else:
                df = pd.DataFrame(results)
                df = df.sort_values("Score", ascending=False).head(12)

                for _, row in df.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([6, 2])

                        with col1:
                            st.markdown(f"**{row['Job Title']}**")
                            st.caption(f"{row['Company']} • {row['Location']}")
                            st.caption(f"{row['Category']}")

                            if row["Matched Skills"]:
                                st.markdown("**Matching skills found in both resume & job description:**")
                                tags = [f"`{s}`" for s in row["Matched Skills"]]
                                st.markdown("  •  ".join(tags))
                            else:
                                st.caption("No direct skill match (rare)")

                        with col2:
                            st.metric("Match", f"{row['Score %']}%")
                            st.progress(float(row["Score"]) / 100)



                        st.caption(f"Salary: {row['Salary']}")
                        with st.expander("Job Description preview"):
                            st.write(row["Description Preview"])

else:

    st.info("Please upload your resume to start matching.")
