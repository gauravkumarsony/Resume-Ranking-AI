import streamlit as st
import base64
import pandas as pd
import numpy as np
import PyPDF2
from io import BytesIO
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')
import textwrap


# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
# Custom background and UI styling
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# CSS for styling based on theme
def get_css():
    if st.session_state.theme == 'dark':
        # Set futuristic theme and UI enhancements
        add_bg_from_local("bg4.jpg")
        return """
        <style>
        .main {
            background-color: #1E1E2D;
            color: #FFFFFF;
        }
        .stTextInput, .stSelectbox {
            background-color: #2D2D3F !important;
            color: #FFFFFF !important;
            border-color: #3D3D4F !important;
        }
        .header {
             border: 1px solid white !important;
             text-align: center;
             padding: 5px;
             border-radius: 20px;
             margin-top: 20px; 
             margin-bottom: 20px;
             background: rgba(255, 255, 255, 0.1);
        }
         .disc p { 
        background-color: #5DA0F6;
        border-radius: 10px;
        padding: 2px;
        }
        .stTextArea {
        background-color: #2D2D3F !important;
            color: #101E36;
            border-radius: 10px;
            padding: 10px;
        }
        div.stTextArea > label {
        color: white !important;  /* Change the color */
        font-weight: bold;
        font-size: 16px;
        }
       .subheader h3{
            color: white;
            margin-top: 20px;
            margin-bottom: 30px !important;
            background-color: #5DA0F6; !important;
            border-radius: 10px;
            padding: 10px;  
        }
        .stButton>button {
            background-color: #5DA0F6;
            color: white;
        }
        .stFileUploader {
            background-color: #2D2D3F !important;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        div.stFileUploader > label {
        color: white !important;  /* Change the color */
        font-weight: bold;
        font-size: 16px;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .card {
            background-color: #2D2D3F;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .card-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        /* Skill Badge Styling */
        .skill-badge {
        display: inline-block;
        background: #5DA0F6;
        color: white;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        }
       /* Styling for Expander */
        details {
            background-color: #252542;
            color: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
            font-size: 16px;
            }
        summary {
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            padding: 10px;
            background-color: #207599;
            color: white;
            border-radius: 5px;
            text-align: left;
        }
        details p {
            font-size: 16px;
            line-height: 1.5;
            margin-top: 10px;
            }
        .match-score {
            font-size: 2rem;
            font-weight: bold;
            color: #5DA0F6;
        }
        .custom-divider {
            border-top: 2px solid white !important; /* White for dark mode */
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .about {
        text-align: center;
        padding: 20px; 
        margin-top: 50px;
        border: 1px solid white !important; 
        background: rgba(255, 255, 255, 0.1); 
        border-radius: 10px;
        }
        </style>
        """
    else:
        # Set futuristic theme and UI enhancements
        add_bg_from_local("bg5.jpg")
        return """
        <style>
        .main {
            background-color: #1E1E2D;
            color: #FFFFFF;
        }
        .stTextInput, .stSelectbox {
                background-color: #F5F5F5 !important;
                border-radius: 8px !important;
                border: 1px solid #E1E1E1 !important;
                color: #101E36 !important;
                transition: all 0.3s ease !important;
        }
        .stTextArea {
        background-color: #2D2D3F !important;
        color: #101E3 !important;
        border-color: #3D3D4F !important;
        border-radius: 10px;
        padding: 10px;
        }
        div.stTextArea > label {
        color: white !important;  /* Change the color */
        font-weight: bold;
        font-size: 16px;
        }
        .stButton>button {
            background-color: #5DA0F6;
            color: white;
        }
        .header {
             border: 1px solid black !important;
             text-align: center;
             padding: 5px;
             border-radius: 20px;
             margin-top: 20px; 
             margin-bottom: 20px;
             background: rgba(0, 0, 0, 0.1);
        }
        .stFileUploader {
            background-color: #2D2D3F !important;
            color: white;
            border-radius: 10px;
            padding: 10px;
            }
        .stMarkdown {
            color: #101E36;
            border-radius: 10px;
            }
        .subheader {
            color: #101E36;
            margin-top: 20px;
            margin-bottom: 10px;  
        }
        .subheader h3{
            color: white;
            margin-top: 20px;
            margin-bottom: 30px !important;
            background-color: #5DA0F6; !important;
            border-radius: 10px;
            padding: 10px;  
        }
        .card {
            background-color: #2D2D3F;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        /* Styling for Expander */
        details {
            background-color: #E3E3E3;
            color: black;
            border-radius: 10px;
            padding: 10px;
            margin-top: 20px !important; 
            margin-bottom: 20px !important; 
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            font-size: 16px;
        }
        summary {
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            padding: 10px;
            background-color: #207599;
            color: white;
            border-radius: 5px;
            text-align: left;
        }
        details p {
            font-size: 16px;
            line-height: 1.5;
            margin-top: 10px;
        }
        .card p {
        display: inline-block;
        background: #5DA0F6;
        color: white;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        } 
        div.stFileUploader > label {
        color: white !important;  /* Change the color */
        font-weight: bold;
        font-size: 16px;
        }
        .card-title {
            font-size: 1.2rem;
            color: white;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .disc p { 
        background-color: #5DA0F6;
        border-radius: 10px;
        padding: 2px;
        }
        .match-score {
            font-size: 2rem;
            font-weight: bold;
            color: #5DA0F6;
        }

        /* Skill Badge Styling */
        .skill-badge {
        display: inline-block;
        background: #5DA0F6;
        color: white;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        }
        .custom-divider {
            border: none;
            background-color: grey !important; 
            width: 100%;
            height: 2px; /* Thicker for better visibility */
            margin: 20px 0;
        }
        .about {
        text-align: center;
        padding: 20px; 
        color: black;
        border: 1px solid black !important;
        margin-top: 40px; 
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        }
        </style>
        """

# Apply CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Function to toggle theme
def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
    

# Title and theme toggle
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
<div class="header">
    <h1>Resume Screening <span style="color: #5DA0F6;">AI</span></h1>
</div>
""", unsafe_allow_html=True)
with col2:
    st.button(
        "🌙" if st.session_state.theme == 'light' else "☀️",
        on_click=toggle_theme,
        key="theme_toggle"
    )

st.markdown("""
<div class="disc">
    <p>Find the perfect candidates effortlessly. Upload job descriptions and resumes to get AI-powered rankings.</p>
</div>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to calculate similarity
def calculate_similarity(job_desc, resume_text):
    corpus = [job_desc, resume_text]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(count_matrix)[0][1]
    return round(similarity * 100, 2)

# Function to extract key skills
def extract_key_skills(text, common_skills):
    skills_found = []
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower()):
            skills_found.append(skill)
    return skills_found

# Common tech skills
common_skills = ['Python', 'JavaScript', 'TypeScript', 'React', 'Node.js', 'Java', 'C++', 'C#', 
    'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'AWS', 'Azure', 'GCP', 
    'Docker', 'Kubernetes', 'Machine Learning', 'AI', 'Data Science', 'Data Analysis',
    'HTML', 'CSS', 'Git', 'TensorFlow', 'PyTorch', 'Django', 'Flask', 'Express',
    'REST API', 'GraphQL', 'DevOps', 'CI/CD', 'Agile', 'Scrum', 'Product Management',
    'Leadership', 'Communication', 'Problem Solving', 'Critical Thinking',
    'Redux', 'Vue.js', 'Angular', 'Spring', 'Hibernate', 'JPA', 'ORM',
    'Unit Testing', 'TDD', 'BDD', 'Jenkins', 'Travis CI', 'CircleCI',
    'Linux', 'Unix', 'Windows', 'Scripting', 'Shell', 'Bash',
    'Frontend', 'Backend', 'Full Stack', 'Mobile Development', 'iOS', 'Android',
    'UX/UI', 'Design', 'Figma', 'Adobe', 'Photoshop', 'Illustrator',
    'Project Management', 'JIRA', 'Confluence', 'Trello', 'Asana',
    'Data Visualization', 'Tableau', 'Power BI', 'D3.js',
    'Big Data', 'Hadoop', 'Spark', 'MapReduce', 'ETL',
    'Networking', 'Security', 'Cryptography', 'Authentication', 'Authorization',
    'Cloud Computing', 'Serverless', 'Lambda', 'Functions', 'Microservices',
    'RESTful', 'SOAP', 'API Design', 'Swagger', 'OpenAPI',
     'MATLAB', 'SAS', 'SPSS', 'Excel', 'VBA',
    'Swift', 'Kotlin', 'Objective-C', 'Flutter', 'React Native',
    'Scala', 'Rust', 'Go', 'Ruby', 'PHP', 'Perl', 'Haskell',
    'Blockchain', 'Smart Contracts', 'Ethereum', 'Solidity',
    'Natural Language Processing', 'Computer Vision', 'Deep Learning',
    'Statistics', 'Probability', 'Linear Algebra', 'Calculus',
    'Database Design', 'Normalization', 'Denormalization', 'Indexing',
    'Load Balancing', 'Caching', 'CDN', 'Performance Optimization',
    'UI/UX', 'Responsive Design', 'Mobile First', 'Accessibility',
    'SEO', 'Google Analytics', 'Marketing', 'Growth Hacking',
    'Leadership', 'Management', 'Team Building', 'Mentoring',
    'Scientific Computing', 'Computational Physics', 'Bioinformatics',
    'Game Development', 'Unity', 'Unreal Engine', 'WebGL',
    'Embedded Systems', 'IoT', 'Robotics', 'Hardware', 'Firmware']


# Job Description
st.markdown("""
<div class="subheader">
    <h2>Job Description</h2>
</div>
""", unsafe_allow_html=True)

job_description = st.text_area("Enter the job description here:", height=200)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
# Resume Upload
st.markdown("""
<div class="subheader">
    <h2>Upload Resumes(PDF)</h2>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload one or more resumes (PDF format)", accept_multiple_files=True, type=['pdf'])

# Process button
process_button = st.button("Rank Resumes")
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

if process_button and job_description and uploaded_files:
    with st.spinner("Analyzing resumes..."):
        processed_jd = preprocess_text(job_description)
        results = []
        
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(BytesIO(file.read()))
            file.seek(0)
            processed_resume = preprocess_text(resume_text)
            match_score = calculate_similarity(processed_jd, processed_resume)
            key_skills = extract_key_skills(resume_text, common_skills)
            results.append({"filename": file.name, "match_score": match_score, "key_skills": key_skills})
        
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        st.markdown("""
            <div class="subheader">
                <h3>Resume Ranking Results</h3>
            </div>
            """, unsafe_allow_html=True)
        
        for i, result in enumerate(results):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i+1}. {result['filename']}**")
                st.progress(result["match_score"] / 100)
                st.markdown(f"Match Score: **{result['match_score']}%**")
            with col2:
                st.markdown("**Key Skills:**")
                if result['key_skills']:
                    skill_tags = "".join([f'<span class="skill-badge">{skill}</span>' for skill in result['key_skills']])
                    st.markdown(skill_tags, unsafe_allow_html=True)
                else:
                    st.markdown("_No matching skills found._")
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Add instructions/help at the bottom
with st.expander("How to use this tool"):
    st.markdown("""
    1. Enter the complete job description in the text area.
    2. Upload one or more resumes in PDF format.
    3. Click the "Rank Resumes" button to process.
    4. Review the results showing match percentages and key skills.
    5. Download the results as a CSV file if needed.
    
    The ranking is based on how well each resume matches the job description using text similarity analysis. The system also extracts key skills from each resume.
    """)

# Footer with additional information
st.markdown("""
    <div class="about">
    <h2>About This Tool</h2>
    <p> 
        Author: Gaurav Kumar <br>
        Technology: NLP, Machine Learning & Scikit-Learn <br>
        Model: Cosine Similarity & Bag-of-Words (BoW)
    </p>
    </div>
""", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("© 2025 Resume Screening AI | By Gaurav Kumar")