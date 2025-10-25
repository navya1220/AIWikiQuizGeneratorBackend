from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
import json
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai

from database import engine, get_db
from models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="DeepKlarity - AI Wiki Quiz Generator",
    description="Generate quizzes from Wikipedia articles using AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizGenerateRequest(BaseModel):
    url: str

class QuizResponse(BaseModel):
    id: int
    url: str
    title: str
    summary: str
    key_entities: dict
    sections: list
    created_at: str

class QuizDetailResponse(QuizResponse):
    quiz_data: dict

class WikipediaScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DeepKlarity-AI-Quiz-Generator/1.0'
        })

    def scrape_article(self, url: str):
        try:
            if not self._is_valid_wikipedia_url(url):
                raise ValueError("Invalid Wikipedia URL")

            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            title = self._extract_title(soup)
            summary = self._extract_summary(soup)
            sections = self._extract_sections(soup)
            full_content = self._extract_full_content(soup)

            return {
                "title": title,
                "summary": summary,
                "sections": sections,
                "key_entities": {"people": [], "organizations": [], "locations": []},
                "full_content": full_content
            }

        except Exception as e:
            logger.error(f"Error scraping Wikipedia article: {e}")
            raise

    def _is_valid_wikipedia_url(self, url: str):
        wikipedia_pattern = r'^https://[a-z]{2}\.wikipedia\.org/wiki/[^/]+$'
        return re.match(wikipedia_pattern, url) is not None

    def _extract_title(self, soup):
        title_element = soup.find('h1', {'class': 'firstHeading'})
        return title_element.get_text().strip() if title_element else "Unknown Title"

    def _extract_summary(self, soup):
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return ""

        paragraphs = content.find_all('p', limit=3)
        summary_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        summary_text = re.sub(r'\[\d+\]', '', summary_text)
        return summary_text[:1000]

    def _extract_full_content(self, soup):
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return ""

        for element in content.find_all(['table', 'div.navbox', 'div.reflist']):
            element.decompose()

        paragraphs = content.find_all('p')
        full_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        full_text = re.sub(r'\[\d+\]', '', full_text)
        return full_text[:8000]

    def _extract_sections(self, soup):
        sections = []
        heading_elements = soup.find_all(['h2', 'h3'])
        
        for heading in heading_elements:
            skip_sections = ['contents', 'references', 'external links', 'see also', 'notes']
            heading_text = heading.get_text().strip().lower()
            
            if not any(skip in heading_text for skip in skip_sections):
                sections.append(heading.get_text().strip())
        
        return sections[:10]

class AdvancedQuizGenerator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            logger.warning("No Google API key found - will use enhanced fallback quizzes")

    def generate_quiz(self, article_content: str, article_title: str = ""):
        """Generate 5-7 unique questions based on actual article content"""
        try:
            if not self.api_key or not self.model:
                logger.info("Using enhanced fallback quiz generator")
                return self._generate_smart_fallback_quiz(article_title)

            if not article_content or len(article_content.strip()) < 100:
                raise ValueError("Article content too short for quiz generation")

            logger.info("Generating AI-powered quiz with Google Gemini...")
            
            # Use first 5000 characters to avoid token limits
            limited_content = article_content[:5000]
            
            prompt = self._create_smart_prompt(limited_content, article_title)
            
            # Generate quiz
            response = self.model.generate_content(prompt)
            quiz_text = response.text
            
            logger.info(f"AI Response received: {len(quiz_text)} characters")
            
            # Clean and parse
            cleaned_text = self._clean_ai_response(quiz_text)
            quiz_data = self._parse_quiz_data(cleaned_text, article_title)
            
            logger.info(f"Successfully generated {len(quiz_data['questions'])} questions")
            return quiz_data
            
        except Exception as e:
            logger.error(f"AI quiz generation failed: {e}")
            return self._generate_smart_fallback_quiz(article_title)

    def _create_smart_prompt(self, content: str, title: str) -> str:
        """Create a detailed prompt for better question generation"""
        return f"""You are an expert quiz creator and educator. Create 5-7 high-quality multiple-choice questions based EXCLUSIVELY on this Wikipedia article.

ARTICLE TITLE: {title}
ARTICLE CONTENT:
{content[:4000]}

CRITICAL INSTRUCTIONS:
1. Generate 5-7 UNIQUE questions that test REAL understanding of the article
2. Each question MUST be directly based on specific facts from the article
3. Questions should cover different aspects: definitions, facts, relationships, applications
4. Make options plausible but only ONE correct based on the article
5. Include varied difficulty levels

REQUIRED FORMAT (JSON only):
{{
  "questions": [
    {{
      "question": "Specific question based on article facts?",
      "options": {{
        "A": "Correct answer from article",
        "B": "Plausible but incorrect alternative",
        "C": "Another incorrect alternative", 
        "D": "Final incorrect alternative"
      }},
      "correct_answer": "A",
      "explanation": "Specific reference to article content explaining why this is correct",
      "difficulty": "easy/medium/hard"
    }}
  ],
  "related_topics": ["SpecificTopic1", "SpecificTopic2", "SpecificTopic3"]
}}

IMPORTANT: Questions MUST be specific to this article, not generic. Focus on unique facts about {title}.
"""

    def _clean_ai_response(self, text: str) -> str:
        """Extract JSON from AI response"""
        # Remove code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        # Find JSON object
        json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL) or re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        
        return text

    def _parse_quiz_data(self, text: str, article_title: str) -> dict:
        """Parse and validate quiz data"""
        try:
            quiz_data = json.loads(text)
            
            # Validate structure
            if "questions" not in quiz_data or not quiz_data["questions"]:
                raise ValueError("No questions in response")
                
            # Ensure minimum 3 questions
            if len(quiz_data["questions"]) < 3:
                logger.warning(f"Only {len(quiz_data['questions'])} questions generated, adding fallback questions")
                fallback = self._generate_smart_fallback_quiz(article_title)
                quiz_data["questions"].extend(fallback["questions"][:5 - len(quiz_data["questions"])])
            
            return self._validate_quiz_structure(quiz_data, article_title)
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return self._generate_smart_fallback_quiz(article_title)

    def _validate_quiz_structure(self, quiz_data: dict, article_title: str) -> dict:
        """Ensure quiz data has proper structure"""
        # Ensure questions exist
        if "questions" not in quiz_data:
            quiz_data["questions"] = []
        
        # Validate each question
        for i, question in enumerate(quiz_data["questions"]):
            # Ensure required fields
            if "question" not in question or not question["question"].strip():
                question["question"] = f"What is a key fact about {article_title}?"
            
            if "options" not in question or len(question["options"]) != 4:
                question["options"] = {
                    "A": f"A key fact about {article_title}",
                    "B": "An incorrect alternative",
                    "C": "Another incorrect option", 
                    "D": "Final incorrect option"
                }
            
            if "correct_answer" not in question or question["correct_answer"] not in ["A", "B", "C", "D"]:
                question["correct_answer"] = "A"
            
            if "explanation" not in question or not question["explanation"].strip():
                question["explanation"] = f"This is correct based on the Wikipedia article about {article_title}."
            
            if "difficulty" not in question or question["difficulty"] not in ["easy", "medium", "hard"]:
                question["difficulty"] = "medium" if i % 2 == 0 else "easy"
        
        # Ensure related topics
        if "related_topics" not in quiz_data or not quiz_data["related_topics"]:
            quiz_data["related_topics"] = [article_title, "Knowledge", "Research"]
        
        return quiz_data

    def _generate_smart_fallback_quiz(self, article_title: str) -> dict:
        """Generate context-aware fallback questions"""
        base_questions = [
            {
                "question": f"What is the primary focus or subject of the Wikipedia article about {article_title}?",
                "options": {
                    "A": f"The main topic: {article_title} and its significance",
                    "B": "A completely unrelated scientific concept",
                    "C": "Historical events from a different time period", 
                    "D": "Fictional stories and characters"
                },
                "correct_answer": "A",
                "explanation": f"The article specifically focuses on {article_title} and provides detailed information about it.",
                "difficulty": "easy"
            },
            {
                "question": f"What type of information would you expect to find in this Wikipedia article about {article_title}?",
                "options": {
                    "A": "Comprehensive facts, history, and context about the subject",
                    "B": "Personal opinions and anecdotes",
                    "C": "Advertising and promotional content",
                    "D": "Fictional narratives and stories"
                },
                "correct_answer": "A",
                "explanation": "Wikipedia articles provide factual, well-researched information with proper citations.",
                "difficulty": "easy"
            },
            {
                "question": f"How does Wikipedia ensure the accuracy of information about topics like {article_title}?",
                "options": {
                    "A": "Through community editing, citations, and reliable sources",
                    "B": "Government verification and approval",
                    "C": "Paid expert reviews only",
                    "D": "Automatic computer generation"
                },
                "correct_answer": "A",
                "explanation": "Wikipedia uses collaborative editing and requires reliable sources to maintain accuracy.",
                "difficulty": "medium"
            },
            {
                "question": f"Why might {article_title} be considered an important topic for a Wikipedia article?",
                "options": {
                    "A": "It represents significant knowledge worth documenting and sharing",
                    "B": "It is trending on social media",
                    "C": "It was randomly selected",
                    "D": "It supports commercial interests"
                },
                "correct_answer": "A",
                "explanation": "Wikipedia documents notable topics that have verifiable significance and reliable sources.",
                "difficulty": "medium"
            },
            {
                "question": f"What makes Wikipedia's coverage of {article_title} valuable for learners and researchers?",
                "options": {
                    "A": "It provides a comprehensive starting point with references for deeper exploration",
                    "B": "It contains all possible information on the topic",
                    "C": "It replaces the need for other information sources",
                    "D": "It offers personalized learning paths"
                },
                "correct_answer": "A",
                "explanation": "Wikipedia offers overviews with citations that help guide further research and learning.",
                "difficulty": "hard"
            }
        ]
        
        return {
            "questions": base_questions,
            "related_topics": [article_title, "Online Encyclopedia", "Knowledge Base"]
        }

@app.get("/")
async def root():
    return {"message": "DeepKlarity AI Wiki Quiz Generator API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/generate_quiz", response_model=QuizDetailResponse)
async def generate_quiz(request: QuizGenerateRequest, db: Session = Depends(get_db)):
    """Generate quiz from Wikipedia URL using AI"""
    try:
        logger.info(f"Generating quiz for: {request.url}")
        
        # Initialize services
        scraper = WikipediaScraper()
        quiz_gen = AdvancedQuizGenerator()

        # Scrape Wikipedia article
        article_data = scraper.scrape_article(request.url)
        logger.info(f"Scraped article: {article_data['title']}")

        # Generate quiz using AI
        quiz_data = quiz_gen.generate_quiz(article_data["full_content"], article_data["title"])
        logger.info(f"Generated {len(quiz_data['questions'])} questions")

        # Store in database
        from models import Quiz
        db_quiz = Quiz(
            url=request.url,
            title=article_data["title"],
            summary=article_data["summary"],
            key_entities=article_data["key_entities"],
            sections=article_data["sections"],
            quiz_data=quiz_data
        )

        db.add(db_quiz)
        db.commit()
        db.refresh(db_quiz)

        return QuizDetailResponse(
            id=db_quiz.id,
            url=db_quiz.url,
            title=db_quiz.title,
            summary=db_quiz.summary,
            key_entities=db_quiz.key_entities,
            sections=db_quiz.sections,
            quiz_data=db_quiz.quiz_data,
            created_at=db_quiz.created_at.isoformat()  # Fixed: convert to string
        )

    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")

@app.get("/api/history", response_model=List[QuizResponse])
async def get_quiz_history(db: Session = Depends(get_db)):
    """Get quiz generation history"""
    from models import Quiz
    try:
        quizzes = db.query(Quiz).order_by(Quiz.created_at.desc()).limit(20).all()
        
        # Convert to proper response format
        response_data = []
        for quiz in quizzes:
            response_data.append({
                "id": quiz.id,
                "url": quiz.url,
                "title": quiz.title,
                "summary": quiz.summary,
                "key_entities": quiz.key_entities or {},
                "sections": quiz.sections or [],
                "created_at": quiz.created_at.isoformat()  # Fixed: convert to string
            })
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quiz history")

@app.get("/api/quiz/{quiz_id}", response_model=QuizDetailResponse)
async def get_quiz_details(quiz_id: int, db: Session = Depends(get_db)):
    """Get full details for a specific quiz"""
    from models import Quiz
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    return QuizDetailResponse(
        id=quiz.id,
        url=quiz.url,
        title=quiz.title,
        summary=quiz.summary,
        key_entities=quiz.key_entities or {},
        sections=quiz.sections or [],
        quiz_data=quiz.quiz_data or {},
        created_at=quiz.created_at.isoformat()  
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)