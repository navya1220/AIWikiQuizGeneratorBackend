import os
import logging
import json
import re
from typing import Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class QuizGenerator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro') if self.api_key else None
        
        self.prompt_template = """You are an expert quiz creator. Using ONLY the Wikipedia article text below, generate 5-7 high-quality multiple-choice questions.

ARTICLE CONTENT:
{article_text}

INSTRUCTIONS:
1. Create 5-7 multiple-choice questions based strictly on the provided article
2. Each question must include:
   - Clear, well-formulated question
   - 4 answer options labeled A, B, C, D
   - Correct answer (just the letter)
   - Concise explanation (1-2 lines)
   - Difficulty level: easy, medium, or hard
3. Include 3 related Wikipedia topics mentioned in the text
4. All answers must be factual and based ONLY on the provided text
5. Questions should test comprehension of key concepts and facts

OUTPUT FORMAT (JSON only):
{
  "questions": [
    {
      "question": "Question text?",
      "options": {
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
      },
      "correct_answer": "A",
      "explanation": "Brief explanation of why this is correct",
      "difficulty": "medium"
    }
  ],
  "related_topics": ["Topic1", "Topic2", "Topic3"]
}

Generate the quiz now:"""

    def generate_quiz(self, article_content: str) -> Dict:
        """Generate quiz from article content using Google Gemini directly"""
        try:
            if not self.api_key:
                logger.warning("No Google API key found, using enhanced fallback quiz")
                return self._create_enhanced_fallback_quiz()

            if not article_content or len(article_content.strip()) < 100:
                raise ValueError("Article content too short for quiz generation")

            logger.info("Generating quiz with Google Gemini...")
            
            # Limit content length to avoid token limits
            limited_content = article_content[:6000]
            
            # Create prompt with article content
            prompt = self.prompt_template.format(article_text=limited_content)
            
            # Generate quiz using Gemini directly
            response = self.model.generate_content(prompt)
            quiz_text = response.text
            
            logger.info(f"Raw AI response: {quiz_text[:200]}...")
            
            # Clean and parse the response
            quiz_text = self._clean_response(quiz_text)
            
            # Parse JSON
            try:
                quiz_data = json.loads(quiz_text)
                logger.info(f"Successfully parsed JSON with {len(quiz_data.get('questions', []))} questions")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Cleaned response: {quiz_text}")
                quiz_data = self._create_enhanced_fallback_quiz()
            
            return self._validate_quiz_data(quiz_data)
            
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return self._create_enhanced_fallback_quiz()

    def _clean_response(self, text: str) -> str:
        """Clean the AI response to extract JSON"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group()
        
        return text.strip()

    def _validate_quiz_data(self, quiz_data: Dict) -> Dict:
        """Validate and ensure quiz data has proper structure"""
        if "questions" not in quiz_data or not quiz_data["questions"]:
            quiz_data["questions"] = self._create_enhanced_fallback_quiz()["questions"]
        
        # Ensure minimum 3 questions
        if len(quiz_data["questions"]) < 3:
            additional_questions = self._create_enhanced_fallback_quiz()["questions"]
            quiz_data["questions"].extend(additional_questions[:3 - len(quiz_data["questions"])])
        
        # Validate each question
        for i, question in enumerate(quiz_data["questions"]):
            if "question" not in question or not question["question"]:
                question["question"] = f"Question about the article content?"
            
            if "options" not in question or len(question["options"]) != 4:
                question["options"] = {
                    "A": "First option based on article",
                    "B": "Second option based on article",
                    "C": "Third option based on article", 
                    "D": "Fourth option based on article"
                }
            
            if "correct_answer" not in question or question["correct_answer"] not in ["A", "B", "C", "D"]:
                question["correct_answer"] = "A"
            
            if "explanation" not in question or not question["explanation"]:
                question["explanation"] = "This answer is correct based on the article content."
            
            if "difficulty" not in question or question["difficulty"] not in ["easy", "medium", "hard"]:
                question["difficulty"] = "medium"
        
        if "related_topics" not in quiz_data or not quiz_data["related_topics"]:
            quiz_data["related_topics"] = ["Knowledge", "Education", "Research"]
        
        return quiz_data

    def _create_enhanced_fallback_quiz(self) -> Dict:
        """Create better fallback quiz with more questions"""
        return {
            "questions": [
                {
                    "question": "What is the primary subject or topic covered in this Wikipedia article?",
                    "options": {
                        "A": "The main subject described in the article",
                        "B": "A completely unrelated scientific field", 
                        "C": "Historical events from a different era",
                        "D": "Fictional characters and stories"
                    },
                    "correct_answer": "A",
                    "explanation": "Wikipedia articles are organized around specific subjects and provide detailed information about them.",
                    "difficulty": "easy"
                },
                {
                    "question": "What characteristic makes Wikipedia a unique online resource?",
                    "options": {
                        "A": "It is collaboratively edited by volunteers worldwide",
                        "B": "It requires expensive subscriptions to access",
                        "C": "It contains only government-approved content",
                        "D": "It is written by a single expert author"
                    },
                    "correct_answer": "A",
                    "explanation": "Wikipedia's collaborative editing model allows continuous improvement by contributors globally.",
                    "difficulty": "medium"
                },
                {
                    "question": "How does Wikipedia ensure the reliability of its content?",
                    "options": {
                        "A": "Through citation requirements and community review",
                        "B": "By having government certification",
                        "C": "Through paid professional editors only",
                        "D": "By limiting edits to academic experts"
                    },
                    "correct_answer": "A",
                    "explanation": "Wikipedia uses citations and community monitoring to maintain content quality and accuracy.",
                    "difficulty": "medium"
                },
                {
                    "question": "What is a key advantage of Wikipedia's open editing model?",
                    "options": {
                        "A": "Rapid updates and corrections to information",
                        "B": "Limited access to prevent errors",
                        "C": "Strict government oversight",
                        "D": "Paid content creation only"
                    },
                    "correct_answer": "A",
                    "explanation": "The open editing allows quick updates when new information becomes available or errors are found.",
                    "difficulty": "hard"
                },
                {
                    "question": "Why is Wikipedia considered a valuable starting point for research?",
                    "options": {
                        "A": "It provides overviews and references to primary sources",
                        "B": "It contains all original research needed",
                        "C": "It is the final authority on all topics",
                        "D": "It replaces the need for other sources"
                    },
                    "correct_answer": "A",
                    "explanation": "Wikipedia offers comprehensive overviews and citations that can lead to more specialized sources.",
                    "difficulty": "easy"
                }
            ],
            "related_topics": ["Online Encyclopedia", "Collaborative Knowledge", "Digital Education"]
        }