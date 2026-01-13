from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import spacy
from collections import Counter
import re
import os

app = Flask(__name__)
app.secret_key = 'day51-resume-skill-extractor-secret-key'

# ============================================
# Skill Categories & Patterns
# ============================================

SKILL_CATEGORIES = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'golang',
        'rust', 'swift', 'kotlin', 'php', 'perl', 'scala', 'r', 'matlab', 'sql', 'html',
        'css', 'sass', 'less', 'bash', 'shell', 'powershell', 'lua', 'dart', 'objective-c'
    ],
    'frameworks': [
        'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
        'django', 'flask', 'fastapi', 'express', 'expressjs', 'nodejs', 'node.js', 'spring',
        'spring boot', 'springboot', '.net', 'dotnet', 'asp.net', 'rails', 'ruby on rails',
        'laravel', 'symfony', 'nextjs', 'next.js', 'nuxt', 'nuxtjs', 'svelte', 'gatsby',
        'bootstrap', 'tailwind', 'tailwindcss', 'material ui', 'chakra ui', 'jquery'
    ],
    'databases': [
        'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'sqlite',
        'oracle', 'sql server', 'mssql', 'mariadb', 'cassandra', 'dynamodb', 'firebase',
        'firestore', 'neo4j', 'couchdb', 'influxdb', 'timescaledb'
    ],
    'cloud_devops': [
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
        'k8s', 'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions', 'circleci',
        'travis ci', 'heroku', 'vercel', 'netlify', 'digitalocean', 'linux', 'nginx', 'apache'
    ],
    'ai_ml': [
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
        'sklearn', 'pandas', 'numpy', 'opencv', 'nlp', 'natural language processing',
        'computer vision', 'neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'bert',
        'gpt', 'langchain', 'huggingface', 'hugging face', 'spacy', 'nltk'
    ],
    'tools': [
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'trello',
        'figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator', 'vs code', 'vscode',
        'intellij', 'pycharm', 'eclipse', 'postman', 'swagger', 'grafana', 'prometheus'
    ],
    'soft_skills': [
        'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
        'critical thinking', 'project management', 'agile', 'scrum', 'kanban',
        'time management', 'presentation', 'mentoring', 'collaboration'
    ]
}

# Flatten all skills for quick lookup
ALL_SKILLS = set()
for skills in SKILL_CATEGORIES.values():
    ALL_SKILLS.update(skills)


# ============================================
# Resume Skill Extractor Engine
# ============================================

class ResumeSkillExtractor:
    """
    Extract skills from resume using pattern matching and NER.
    Summarize findings using HuggingFace LLM.
    """
    
    def __init__(self):
        self.nlp = None
        self.summarizer = None
        self.is_ready = False
        self.stats = {
            'resumes_processed': 0,
            'skills_extracted': 0
        }
    
    def initialize(self):
        """Initialize spaCy and HuggingFace models."""
        try:
            print("🔧 Loading spaCy model...")
            
            # Try to load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("📥 Downloading spaCy model...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            print("✅ spaCy loaded!")
            
            print("🔧 Loading summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=50,
                do_sample=False
            )
            print("✅ Summarizer loaded!")
            
            # Test
            test_skills = self.extract_skills("I know Python, React, and AWS")
            if test_skills:
                print(f"✅ Test passed: Found {len(test_skills['all_skills'])} skills")
            
            self.is_ready = True
            print("✅ Resume Skill Extractor initialized successfully!")
            return True, "Models loaded successfully!"
            
        except Exception as e:
            print(f"❌ Error initializing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Error: {str(e)}"
    
    def extract_skills(self, text):
        """
        Extract skills from resume text.
        
        Uses:
        1. Pattern matching for known skills
        2. spaCy NER for additional entities
        """
        if not text or not text.strip():
            return None
        
        text_lower = text.lower()
        found_skills = {category: [] for category in SKILL_CATEGORIES}
        
        # Pattern matching for known skills
        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                # Use word boundary matching
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    # Get original case from text
                    match = re.search(pattern, text_lower)
                    if match:
                        original_skill = text[match.start():match.end()]
                        if skill not in [s.lower() for s in found_skills[category]]:
                            found_skills[category].append(original_skill)
        
        # Use spaCy NER for additional entities
        entities = []
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_
                    })
        
        # Compile all skills
        all_skills = []
        for category, skills in found_skills.items():
            all_skills.extend(skills)
        
        # Count by category
        category_counts = {
            cat: len(skills) for cat, skills in found_skills.items() if skills
        }
        
        return {
            'skills_by_category': found_skills,
            'all_skills': all_skills,
            'category_counts': category_counts,
            'entities': entities[:10],  # Limit entities
            'total_skills': len(all_skills)
        }
    
    def generate_summary(self, text, skills_data):
        """Generate a professional summary using the extracted skills."""
        if not self.summarizer:
            return self._fallback_summary(skills_data)
        
        try:
            # Create a skills overview
            skills_text = self._create_skills_overview(skills_data)
            
            # Combine with resume text for context
            combined_text = f"""
            Resume Analysis:
            {text[:1500]}
            
            Key Skills Identified:
            {skills_text}
            """
            
            # Generate summary
            if len(combined_text.split()) > 50:
                result = self.summarizer(combined_text[:2000], max_length=150, min_length=40)
                generated_summary = result[0]['summary_text']
            else:
                generated_summary = None
            
            # Create structured summary
            summary = self._create_structured_summary(skills_data, generated_summary)
            
            return summary
            
        except Exception as e:
            print(f"Summarization error: {e}")
            return self._fallback_summary(skills_data)
    
    def _create_skills_overview(self, skills_data):
        """Create a text overview of skills."""
        lines = []
        for category, skills in skills_data['skills_by_category'].items():
            if skills:
                category_name = category.replace('_', ' ').title()
                lines.append(f"{category_name}: {', '.join(skills)}")
        return '\n'.join(lines)
    
    def _create_structured_summary(self, skills_data, generated_summary=None):
        """Create a structured professional summary."""
        total = skills_data['total_skills']
        categories = skills_data['category_counts']
        
        # Determine profile type
        profile_type = self._determine_profile_type(categories)
        
        # Build summary parts
        summary_parts = []
        
        # Opening
        if total > 0:
            summary_parts.append(
                f"This candidate demonstrates expertise across {len(categories)} technical domains "
                f"with {total} identified skills."
            )
        
        # Profile type
        if profile_type:
            summary_parts.append(f"Profile Type: {profile_type}")
        
        # Top categories
        if categories:
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            top_areas = [cat.replace('_', ' ').title() for cat, _ in sorted_cats]
            summary_parts.append(f"Primary expertise in: {', '.join(top_areas)}")
        
        # AI/ML mention
        if categories.get('ai_ml', 0) > 0:
            summary_parts.append("Notable AI/ML background detected.")
        
        # Generated insight
        if generated_summary:
            summary_parts.append(f"\nAI Analysis: {generated_summary}")
        
        return ' '.join(summary_parts)
    
    def _determine_profile_type(self, categories):
        """Determine the candidate's profile type based on skills."""
        if not categories:
            return "General"
        
        # Calculate scores
        frontend_score = categories.get('frameworks', 0)
        backend_score = categories.get('programming_languages', 0) + categories.get('databases', 0)
        devops_score = categories.get('cloud_devops', 0)
        ml_score = categories.get('ai_ml', 0)
        
        max_score = max(frontend_score, backend_score, devops_score, ml_score)
        
        if max_score == 0:
            return "General Professional"
        
        if ml_score == max_score and ml_score >= 2:
            return "AI/ML Engineer"
        elif devops_score == max_score and devops_score >= 2:
            return "DevOps/Cloud Engineer"
        elif frontend_score >= 2 and backend_score >= 2:
            return "Full Stack Developer"
        elif frontend_score == max_score:
            return "Frontend Developer"
        elif backend_score == max_score:
            return "Backend Developer"
        else:
            return "Software Developer"
    
    def _fallback_summary(self, skills_data):
        """Create a summary without the LLM."""
        total = skills_data['total_skills']
        categories = skills_data['category_counts']
        
        if total == 0:
            return "No technical skills detected in the provided text."
        
        profile_type = self._determine_profile_type(categories)
        
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        top_areas = [f"{cat.replace('_', ' ').title()} ({count})" for cat, count in sorted_cats[:3]]
        
        return (
            f"Candidate Profile: {profile_type}. "
            f"Identified {total} skills across {len(categories)} categories. "
            f"Top areas: {', '.join(top_areas)}."
        )
    
    def process_resume(self, text):
        """Full resume processing pipeline."""
        # Extract skills
        skills_data = self.extract_skills(text)
        
        if not skills_data:
            return None
        
        # Generate summary
        summary = self.generate_summary(text, skills_data)
        
        # Update stats
        self.stats['resumes_processed'] += 1
        self.stats['skills_extracted'] += skills_data['total_skills']
        
        return {
            'skills': skills_data,
            'summary': summary,
            'profile_type': self._determine_profile_type(skills_data['category_counts'])
        }
    
    def get_stats(self):
        """Get processing statistics."""
        return {
            'is_ready': self.is_ready,
            'resumes_processed': self.stats['resumes_processed'],
            'skills_extracted': self.stats['skills_extracted']
        }


# Initialize extractor
extractor = ResumeSkillExtractor()


# ============================================
# Flask Routes
# ============================================

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract_skills():
    """Extract skills from resume text."""
    if not extractor.is_ready:
        return jsonify({
            'success': False,
            'error': 'Models are still loading. Please wait...'
        })
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({
            'success': False,
            'error': 'Please provide resume text.'
        })
    
    try:
        result = extractor.process_resume(text)
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Could not process the resume.'
            })
        
        return jsonify({
            'success': True,
            'skills': result['skills'],
            'summary': result['summary'],
            'profile_type': result['profile_type']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/status')
def status():
    """Check if models are ready."""
    return jsonify({
        'ready': extractor.is_ready,
        'stats': extractor.get_stats()
    })


@app.route('/sample')
def sample():
    """Get a sample resume."""
    sample_resume = """
    John Smith
    Senior Software Engineer
    
    Summary:
    Experienced software engineer with 8+ years of expertise in building scalable web applications 
    and cloud-native solutions. Strong background in Python, JavaScript, and cloud technologies.
    
    Technical Skills:
    - Programming: Python, JavaScript, TypeScript, Java, SQL
    - Frontend: React, Vue.js, HTML5, CSS3, Tailwind CSS
    - Backend: Django, Flask, Node.js, Express, FastAPI
    - Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
    - Cloud & DevOps: AWS, Docker, Kubernetes, Terraform, CI/CD
    - AI/ML: TensorFlow, PyTorch, scikit-learn, NLP, LangChain
    - Tools: Git, GitHub, Jira, VS Code, Postman
    
    Experience:
    
    Tech Corp Inc. | Senior Software Engineer | 2020 - Present
    - Led development of microservices architecture using Python and Docker
    - Implemented machine learning pipelines with TensorFlow and AWS SageMaker
    - Mentored junior developers and conducted code reviews
    
    StartupXYZ | Full Stack Developer | 2017 - 2020
    - Built React-based dashboard with real-time data visualization
    - Developed REST APIs using Django and PostgreSQL
    - Deployed applications on AWS using Docker and Kubernetes
    
    Education:
    MS Computer Science - Stanford University
    BS Computer Science - MIT
    
    Certifications:
    - AWS Solutions Architect
    - Google Cloud Professional
    - Kubernetes Administrator (CKA)
    """
    return jsonify({'sample': sample_resume.strip()})


# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         Day 51: Resume Skill Extractor                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  NER Engine: spaCy                                           ║
    ║  Summarizer: BART (facebook/bart-large-cnn)                  ║
    ║  Features: Skill Extraction + AI Summary                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting Resume Skill Extractor...")
    success, message = extractor.initialize()
    
    if success:
        print(f"\n🌐 Starting server at http://localhost:5000\n")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    else:
        print(f"❌ Failed to initialize: {message}")
