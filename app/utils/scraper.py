import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class WikipediaScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DeepKlarity-AI-Quiz-Generator/1.0'
        })

    def scrape_article(self, url: str) -> Dict:
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

    def _is_valid_wikipedia_url(self, url: str) -> bool:
        wikipedia_pattern = r'^https://[a-z]{2}\.wikipedia\.org/wiki/[^/]+$'
        return re.match(wikipedia_pattern, url) is not None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_element = soup.find('h1', {'class': 'firstHeading'})
        return title_element.get_text().strip() if title_element else "Unknown Title"

    def _extract_summary(self, soup: BeautifulSoup) -> str:
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return ""

        paragraphs = content.find_all('p', limit=3)
        summary_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        summary_text = re.sub(r'\[\d+\]', '', summary_text)
        return summary_text[:1000]

    def _extract_full_content(self, soup: BeautifulSoup) -> str:
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return ""

        for element in content.find_all(['table', 'div.navbox', 'div.reflist']):
            element.decompose()

        paragraphs = content.find_all('p')
        full_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        full_text = re.sub(r'\[\d+\]', '', full_text)
        return full_text[:8000]

    def _extract_sections(self, soup: BeautifulSoup) -> List[str]:
        sections = []
        heading_elements = soup.find_all(['h2', 'h3'])
        
        for heading in heading_elements:
            skip_sections = ['contents', 'references', 'external links', 'see also', 'notes']
            heading_text = heading.get_text().strip().lower()
            
            if not any(skip in heading_text for skip in skip_sections):
                sections.append(heading.get_text().strip())
        
        return sections[:10]