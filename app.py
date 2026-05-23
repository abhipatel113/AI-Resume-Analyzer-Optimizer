import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
import json

# 1. Initialize Gemini Client
# Ensure GEMINI_API_KEY is set in your environment variables
try:
    # This reads the key from the server environment, keeping it invisible to users
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Gemini API Client failed to initialize. Error: {str(e)}")

# 2. Helper Function: Extract Text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# 3. Gemini Prompt Functions
def analyze_resume_ats(resume_text, job_description):
    """Calculates ATS score and extracts missing keywords using structured JSON."""
    
    prompt = f"""
    You are an expert ATS (Applicant Tracking System) scanner and corporate recruiter.
    Analyze the provided Resume against the Job Description.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide your analysis strictly in the following JSON format:
    {{
        "ats_score": <int between 0 and 100>,
        "match_explanation": "<2-3 sentences summarizing the overall match>",
        "missing_keywords": ["keyword1", "keyword2", "keyword3"],
        "critical_gaps": ["gap1", "gap2"]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Failed to analyze. Error: {str(e)}"}

def improve_bullet_points(resume_text, target_role):
    """Rewrites weak resume bullets into high-impact STAR formatted sentences."""
    
    prompt = f"""
    You are a professional resume writer. Review the following resume text and identify 3-4 weak or average experience bullet points.
    Rewrite them using the STAR method (Situation, Task, Action, Result), ensuring they start with strong action verbs and emphasize quantifiable metrics.
    Target Role: {target_role}
    
    Resume Content:
    {resume_text}
    
    Provide your suggestions strictly in the following JSON format:
    {{
        "improvements": [
            {{
                "original": "Original weak bullet point",
                "improved": "Rewritten impactful STAR bullet point",
                "why": "Explanation of why this is better"
            }}
        ]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Failed to optimize bullets. Error: {str(e)}"}


# 4. Streamlit UI Layout
st.set_page_config(page_title="AI Resume Optimizer", page_icon="📝", layout="wide")

st.title("📝 AI Resume Analyzer & ATS Optimizer")
st.caption("Powered by Gemini 2.5 • Maximize your interview callbacks")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Step 1: Upload Inputs")
    uploaded_file = st.file_uploader("Upload your Resume (PDF format)", type=["pdf"])
    job_desc = st.text_area("Paste the Target Job Description", height=250, placeholder="Paste the full job requirements here...")
    target_role = st.text_input("Target Job Title (e.g., Senior Software Engineer)", placeholder="Software Engineer")

with col2:
    st.subheader("📊 Step 2: Optimization Report")
    
    if not uploaded_file or not job_desc:
        st.info("Upload your resume and paste a job description to trigger the AI analysis.")
    else:
        if st.button("🚀 Run ATS Analysis", use_container_width=True):
            with st.spinner("Gemini is parsing and analyzing your compatibility..."):
                
                # Extract Text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # Run ATS Scoring & Keywords
                ats_result = analyze_resume_ats(resume_text, job_desc)
                
                if "error" in ats_result:
                    st.error(ats_result["error"])
                else:
                    # Metrics Display
                    score = ats_result.get("ats_score", 0)
                    if score >= 80:
                        st.success(f"### ATS Match Score: {score}% 🎉")
                    elif score >= 50:
                        st.warning(f"### ATS Match Score: {score}% ⚠️")
                    else:
                        st.error(f"### ATS Match Score: {score}% 🚨")
                    
                    st.markdown(f"**Analysis:** {ats_result.get('match_explanation')}")
                    
                    st.write("---")
                    
                    # Keywords & Gaps Layout
                    k_col1, k_col2 = st.columns(2)
                    with k_col1:
                        st.markdown("#### 🔍 Missing Key Terms")
                        for kw in ats_result.get("missing_keywords", []):
                            st.caption(f"❌ {kw}")
                            
                    with k_col2:
                        st.markdown("#### ⚠️ Foundational Gaps")
                        for gap in ats_result.get("critical_gaps", []):
                            st.caption(f"👉 {gap}")
                            
            st.write("---")
            
            # Run Bullet Point Enhancer
            with st.spinner("Rewriting weak bullet points into high-impact STAR verbs..."):
                bullet_result = improve_bullet_points(resume_text, target_role if target_role else "Target Role")
                
                if "error" in bullet_result:
                    st.error(bullet_result["error"])
                else:
                    st.markdown("### 📈 AI Bullet Point Enhancements (STAR Method)")
                    for item in bullet_result.get("improvements", []):
                        with st.expander(f"💡 Original: {item.get('original')[:60]}..."):
                            st.markdown(f"**❌ Original:**  \n*{item.get('original')}*")
                            st.markdown(f"**✅ Improved (STAR Format):**  \n**{item.get('improved')}**")
                            st.markdown(f"💡 *Why it works:* {item.get('why')}")