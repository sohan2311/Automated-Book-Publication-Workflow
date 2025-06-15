# Main Execution File

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

import gradio as gr
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentStatus(Enum):
    SCRAPED = "scraped"
    AI_WRITTEN = "ai_written"
    AI_REVIEWED = "ai_reviewed"
    HUMAN_EDITED = "human_edited"
    FINALIZED = "finalized"

@dataclass
class ContentVersion:
    id: str
    content: str
    status: ContentStatus
    timestamp: datetime
    metadata: Dict
    quality_score: float = 0.0
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data

class AIAgent:
    """Base class for AI agents with different roles"""
    
    def __init__(self, role: str, model_name: str = "gemini-pro"):
        self.role = role
        self.model_name = model_name
        

        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key or gemini_key == "your_gemini_api_key_here":
            logger.warning("Gemini API key not found. AI processing will be limited.")
            self.model = None
        else:
            try:
                genai.configure(api_key=gemini_key)
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.model = None
    
    def process_content(self, content: str, context: str = "") -> str:
        """Process content based on agent role - now synchronous"""
        if not self.model:
            logger.warning(f"AI {self.role} not available. Returning original content.")
            return content
            
        prompt = self._get_role_specific_prompt(content, context)
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error in {self.role} processing: {e}")
            return content 
    
    def _get_role_specific_prompt(self, content: str, context: str) -> str:
        prompts = {
            "writer": f"""As an AI Writer, rewrite and enhance this content for better clarity, engagement, and readability.
            Maintain the original meaning while improving flow and style.
            Context: {context}
            
            Original Content:
            {content}
            
            Enhanced Version:""",
            
            "reviewer": f"""As an AI Reviewer, analyze this content and provide an improved version that addresses:
            1. Clarity and coherence
            2. Grammar and style
            3. Factual accuracy
            4. Engagement level
            
            Context: {context}
            
            Content to Review:
            {content}
            
            Improved Version:""",
            
            "editor": f"""As an AI Editor, perform final editing on this content:
            1. Fix any remaining grammatical issues
            2. Ensure consistent tone and style
            3. Optimize for readability
            4. Make final refinements
            
            Context: {context}
            
            Content to Edit:
            {content}
            
            Final Edited Version:"""
        }
        return prompts.get(self.role, content)

class RLSearchAlgorithm:
    """Reinforcement Learning-based search algorithm for content retrieval"""
    
    def __init__(self):
        self.search_history = []
        self.quality_feedback = {}
        self.learning_rate = 0.1
        
    def search_and_rank(self, query: str, candidates: List[ContentVersion]) -> List[ContentVersion]:
        """Search and rank content using RL-enhanced scoring"""
        if not candidates:
            return []
        

        scores = []
        for candidate in candidates:
            base_score = self._calculate_base_relevance(query, candidate)
            rl_adjustment = self._get_rl_adjustment(candidate)
            final_score = base_score + rl_adjustment
            scores.append((final_score, candidate))

        scores.sort(key=lambda x: x[0], reverse=True)
        ranked_candidates = [candidate for _, candidate in scores]
        

        self.search_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'results': [c.id for c in ranked_candidates[:5]]
        })
        
        return ranked_candidates
    
    def _calculate_base_relevance(self, query: str, candidate: ContentVersion) -> float:
        """Calculate base relevance score"""
        query_lower = query.lower()
        content_lower = candidate.content.lower()
        

        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if not query_words:
            return candidate.quality_score
        
        intersection = query_words.intersection(content_words)
        relevance = len(intersection) / len(query_words)
        

        return (relevance * 0.7) + (candidate.quality_score * 0.3)
    
    def _get_rl_adjustment(self, candidate: ContentVersion) -> float:
        """Get RL-based adjustment for ranking"""
        candidate_id = candidate.id
        if candidate_id in self.quality_feedback:
            return self.quality_feedback[candidate_id] * 0.1
        return 0.0
    
    def provide_feedback(self, content_id: str, quality_rating: float):
        """Provide feedback for RL learning"""
        if content_id not in self.quality_feedback:
            self.quality_feedback[content_id] = 0.0
        

        current_score = self.quality_feedback[content_id]
        self.quality_feedback[content_id] = current_score + self.learning_rate * (quality_rating - current_score)

class EnhancedBookWorkflow:
    """Main workflow management class"""
    
    def __init__(self):
        self.setup_chromadb()
        self.ai_writer = AIAgent("writer")
        self.ai_reviewer = AIAgent("reviewer")
        self.ai_editor = AIAgent("editor")
        self.rl_search = RLSearchAlgorithm()
        self.content_versions = {}
        
    def setup_chromadb(self):
        """Initialize ChromaDB with embedding function"""
        try:
            self.chroma_client = chromadb.Client()

            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="book_content_versions",
                embedding_function=self.embedding_function
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"ChromaDB setup error: {e}")
            # Fallback to in-memory storage
            self.collection = None
    
    def scrape_content(self, url: str) -> Tuple[str, str]:
        """Scrape content and take screenshot - now synchronous"""
        try:

            return asyncio.run(self._async_scrape_content(url))
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return "", ""
    
    async def _async_scrape_content(self, url: str) -> Tuple[str, str]:
        """Async scraping implementation"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                logger.info(f"Navigating to: {url}")
                await page.goto(url, timeout=30000)
                

                await page.wait_for_load_state('networkidle', timeout=10000)
                

                content = await page.content()
                

                screenshot_dir = os.path.join(os.getcwd(), "screenshots")
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshot_dir, f"chapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                
                await browser.close()
                

                soup = BeautifulSoup(content, 'html.parser')
                

                for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
                    element.decompose()
                

                paragraphs = soup.find_all('p')
                clean_text = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                
                if not clean_text:

                    clean_text = soup.get_text()

                    clean_text = "\n".join([line.strip() for line in clean_text.split('\n') if line.strip()])
                

                version_id = self._generate_version_id(clean_text, ContentStatus.SCRAPED)
                version = ContentVersion(
                    id=version_id,
                    content=clean_text,
                    status=ContentStatus.SCRAPED,
                    timestamp=datetime.now(),
                    metadata={
                        'source_url': url,
                        'screenshot_path': screenshot_path,
                        'word_count': len(clean_text.split())
                    }
                )
                
                self._save_version(version)
                logger.info(f"Successfully scraped content from {url}")
                return clean_text, screenshot_path
                
        except Exception as e:
            logger.error(f"Async scraping error: {e}")
            return "", ""
    
    def ai_writing_pipeline(self, content: str, iterations: int = 1) -> str:
        """AI writing pipeline with multiple iterations - now synchronous"""
        if not content.strip():
            return "No content provided for processing"
            
        current_content = content
        
        for i in range(iterations):
            logger.info(f"AI Writing iteration {i+1}/{iterations}")
            

            written_content = self.ai_writer.process_content(
                current_content, 
                f"Writing iteration {i+1}, focus on clarity and engagement"
            )
            

            reviewed_content = self.ai_reviewer.process_content(
                written_content,
                f"Review iteration {i+1}, provide constructive feedback"
            )
            
            current_content = reviewed_content
            

            version_id = self._generate_version_id(current_content, ContentStatus.AI_REVIEWED)
            version = ContentVersion(
                id=version_id,
                content=current_content,
                status=ContentStatus.AI_REVIEWED,
                timestamp=datetime.now(),
                metadata={
                    'iteration': i+1,
                    'total_iterations': iterations,
                    'word_count': len(current_content.split())
                },
                quality_score=self._calculate_quality_score(current_content)
            )
            
            self._save_version(version)
        
        return current_content
    
    def human_editing_interface(self, content: str, editor_notes: str = "") -> str:
        """Process human editing input"""
        if not content.strip():
            return "No content provided for editing"
            

        version_id = self._generate_version_id(content, ContentStatus.HUMAN_EDITED)
        version = ContentVersion(
            id=version_id,
            content=content,
            status=ContentStatus.HUMAN_EDITED,
            timestamp=datetime.now(),
            metadata={
                'editor_notes': editor_notes,
                'word_count': len(content.split())
            },
            quality_score=self._calculate_quality_score(content)
        )
        
        self._save_version(version)
        return "Human edit saved successfully!"
    
    def finalize_content(self, content: str) -> str:
        """Final AI editing pass"""
        if not content.strip():
            return "No content provided for finalization"
            
        final_content = self.ai_editor.process_content(
            content,
            "Final editing pass - ensure publication quality"
        )
        
        version_id = self._generate_version_id(final_content, ContentStatus.FINALIZED)
        version = ContentVersion(
            id=version_id,
            content=final_content,
            status=ContentStatus.FINALIZED,
            timestamp=datetime.now(),
            metadata={
                'final_version': True,
                'word_count': len(final_content.split())
            },
            quality_score=self._calculate_quality_score(final_content)
        )
        
        self._save_version(version)
        return final_content
    
    def search_versions(self, query: str, status_filter: Optional[ContentStatus] = None) -> List[ContentVersion]:
        """Search versions using RL algorithm"""
        try:
            if self.collection is None:

                candidates = list(self.content_versions.values())
            else:

                results = self.collection.query(
                    query_texts=[query],
                    n_results=50
                )
                
                candidates = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        version = ContentVersion(
                            id=results['ids'][0][i],
                            content=doc,
                            status=ContentStatus(metadata.get('status', 'scraped')),
                            timestamp=datetime.fromisoformat(metadata['timestamp']),
                            metadata=metadata,
                            quality_score=metadata.get('quality_score', 0.0)
                        )
                        candidates.append(version)
            

            if status_filter:
                candidates = [c for c in candidates if c.status == status_filter]
            

            return self.rl_search.search_and_rank(query, candidates)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _generate_version_id(self, content: str, status: ContentStatus) -> str:
        """Generate unique version ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{status.value}_{timestamp}_{content_hash}"
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate content quality score"""

        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        

        avg_sentence_length = word_count / max(sentence_count, 1)
        

        word_score = min(word_count / 1000, 1.0)
        sentence_score = max(0, 1.0 - abs(avg_sentence_length - 15) / 15)
        
        return (word_score * 0.3 + sentence_score * 0.7)
    
    def _save_version(self, version: ContentVersion):
        """Save version to ChromaDB and memory"""
        self.content_versions[version.id] = version
        
        if self.collection is not None:
            try:
                self.collection.add(
                    documents=[version.content],
                    metadatas=[version.to_dict()],
                    ids=[version.id]
                )
            except Exception as e:
                logger.error(f"ChromaDB save error: {e}")


workflow = EnhancedBookWorkflow()


def scrape_and_extract(url):
    """Gradio function for scraping"""
    if not url:
        url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    
    content, screenshot_path = workflow.scrape_content(url)
    return content, screenshot_path if screenshot_path and os.path.exists(screenshot_path) else None

def ai_process_content(content, iterations):
    """Gradio function for AI processing"""
    if not content.strip():
        return "No content to process"
    
    processed_content = workflow.ai_writing_pipeline(content, max(1, iterations))
    return processed_content

def save_human_edit(content, notes):
    """Gradio function for human editing"""
    if not content.strip():
        return "No content to save", content
    
    result = workflow.human_editing_interface(content, notes)
    return result, content

def finalize_version(content):
    """Gradio function for finalization"""
    if not content.strip():
        return "No content to finalize"
    
    final_content = workflow.finalize_content(content)
    return final_content

def search_content(query, status_filter):
    """Gradio function for searching"""
    if not query.strip():
        return "Please enter a search query"
    
    status = None
    if status_filter != "All":
        status = ContentStatus(status_filter.lower().replace(" ", "_"))
    
    results = workflow.search_versions(query, status)
    
    if not results:
        return "No results found"
    
    output = ""
    for i, result in enumerate(results[:5], 1):
        output += f"**Result {i}:**\n"
        output += f"ID: {result.id}\n"
        output += f"Status: {result.status.value}\n"
        output += f"Quality Score: {result.quality_score:.2f}\n"
        output += f"Timestamp: {result.timestamp}\n"
        output += f"Content Preview: {result.content[:200]}...\n"
        output += "-" * 50 + "\n\n"
    
    return output

def provide_quality_feedback(content_id, rating):
    """Gradio function for quality feedback"""
    if not content_id.strip():
        return "Please enter a content ID"
    
    try:
        rating_float = float(rating)
        workflow.rl_search.provide_feedback(content_id, rating_float)
        return f"Feedback recorded: {rating_float}/5.0 for content {content_id}"
    except ValueError:
        return "Please enter a valid rating (0-5)"

def get_workflow_statistics():
    """Get comprehensive workflow statistics"""
    try:
        versions = list(workflow.content_versions.values())
        
        if not versions:
            return (
                0, {}, 0.0, "No activity yet"
            )
        

        total = len(versions)
        avg_quality = sum(v.quality_score for v in versions) / total
        

        status_counts = {}
        for version in versions:
            status = version.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        

        recent_versions = sorted(versions, key=lambda v: v.timestamp, reverse=True)[:10]
        activity_text = ""
        for v in recent_versions:
            timestamp_str = v.timestamp.strftime("%Y-%m-%d %H:%M")
            activity_text += f"{timestamp_str} - {v.status.value} - Quality: {v.quality_score:.2f}\n"
        
        return (
            total,
            status_counts,
            round(avg_quality, 3),
            activity_text
        )
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return (
            0, {}, 0.0, f"Error loading statistics: {str(e)}"
        )

def export_content(content):
    """Export final content to file"""
    if not content.strip():
        return "No content to export"
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"final_content_{timestamp}.txt"
    
    try:
        export_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ Content exported to: {filepath}"
    except Exception as e:
        return f"❌ Export failed: {str(e)}"


def create_gradio_interface():
    with gr.Blocks(title="Book Publication Workflow", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 📚 Automated Book Publication Workflow
        
        **Seamless AI-Driven Content Processing Pipeline**
        
        ✨ **Pipeline Features:**
        - 🔄 Automatic data flow between stages
        - ⚡ One-click progression through workflow
        - 🎯 Streamlined user experience
        - 📊 Real-time content tracking
        """)
        

        pipeline_content = gr.State("")
        
        with gr.Tab("🌐 1. Content Scraping") as tab1:
            gr.Markdown("### Extract content from web sources")
            
            with gr.Row():
                url_input = gr.Textbox(
                    label="Source URL",
                    value="https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1",
                    placeholder="Enter URL to scrape"
                )
            
            scrape_btn = gr.Button("🔍 Extract Content", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    scraped_content = gr.Textbox(
                        label="Extracted Content",
                        lines=15,
                        interactive=True
                    )
                    

                    with gr.Row():
                        send_to_ai_btn = gr.Button("➡️ Send to AI Processing", variant="secondary", size="lg")
                        content_status = gr.Textbox(
                            label="Status",
                            value="Ready for extraction",
                            interactive=False,
                            lines=1
                        )
                
                with gr.Column():
                    screenshot_output = gr.Image(
                        label="Page Screenshot",
                        type="filepath"
                    )
        
        with gr.Tab("🤖 2. AI Processing") as tab2:
            gr.Markdown("### AI enhancement and refinement")
            
            with gr.Row():
                ai_iterations = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="AI Processing Iterations"
                )
                process_ai_btn = gr.Button("🚀 Process with AI", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    ai_input_display = gr.Textbox(
                        label="Content from Previous Stage",
                        lines=8,
                        interactive=False
                    )
                    
                with gr.Column():
                    ai_output_content = gr.Textbox(
                        label="AI Processed Content",
                        lines=12,
                        interactive=True
                    )
            
            with gr.Row():
                send_to_editing_btn = gr.Button("➡️ Send to Human Editing", variant="secondary", size="lg")
                skip_to_final_btn = gr.Button("⏭️ Skip to Final Processing", variant="outline", size="lg")
                ai_status = gr.Textbox(
                    label="Status",
                    value="Waiting for content...",
                    interactive=False,
                    lines=1
                )
        
        with gr.Tab("✏️ 3. Human Editing") as tab3:
            gr.Markdown("### Manual review and editing")
            
            with gr.Row():
                with gr.Column():
                    human_input_display = gr.Textbox(
                        label="Content from AI Processing",
                        lines=8,
                        interactive=False
                    )
                    
                    editor_notes = gr.Textbox(
                        label="Editor Notes",
                        lines=3,
                        placeholder="Add your editing notes and comments"
                    )
                    
                with gr.Column():
                    human_edit_content = gr.Textbox(
                        label="Edit Content Here",
                        lines=15,
                        interactive=True
                    )
            
            with gr.Row():
                save_edit_btn = gr.Button("💾 Save Edits", variant="primary", size="lg")
                send_to_final_btn = gr.Button("➡️ Send to Final Processing", variant="secondary", size="lg")
                edit_status = gr.Textbox(
                    label="Status",
                    value="Waiting for content...",
                    interactive=False,
                    lines=1
                )
        
        with gr.Tab("🏁 4. Final Processing") as tab4:
            gr.Markdown("### Final AI polish and publication preparation")
            
            with gr.Row():
                with gr.Column():
                    final_input_display = gr.Textbox(
                        label="Content from Previous Stage",
                        lines=8,
                        interactive=False
                    )
                    
                    finalize_btn = gr.Button("🏁 Finalize Content", variant="primary", size="lg")
                    
                with gr.Column():
                    final_output_content = gr.Textbox(
                        label="Publication-Ready Content",
                        lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                export_btn = gr.Button("📥 Export Content", variant="secondary", size="lg")
                restart_btn = gr.Button("🔄 Start New Workflow", variant="outline", size="lg")
                final_status = gr.Textbox(
                    label="Status",
                    value="Waiting for content...",
                    interactive=False,
                    lines=1
                )
        
        with gr.Tab("🔍 5. Search & Analytics"):
            gr.Markdown("### Search through content versions and provide quality feedback")
            
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter keywords to search for"
                    )
                    status_filter = gr.Dropdown(
                        choices=["All", "Scraped", "AI Written", "AI Reviewed", "Human Edited", "Finalized"],
                        value="All",
                        label="Filter by Status"
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary")
                
                with gr.Column():
                    feedback_content_id = gr.Textbox(
                        label="Content ID for Feedback",
                        placeholder="Enter content ID from search results"
                    )
                    quality_rating = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=3,
                        step=0.1,
                        label="Quality Rating (0-5)"
                    )
                    feedback_btn = gr.Button("📊 Submit Feedback", variant="secondary")
            
            with gr.Row():
                search_results = gr.Textbox(
                    label="Search Results",
                    lines=20,
                    interactive=False
                )
            
            with gr.Row():
                feedback_status = gr.Textbox(
                    label="Feedback Status",
                    lines=2,
                    interactive=False
                )
        
        with gr.Tab("📊 6. Workflow Statistics"):
            gr.Markdown("### View workflow statistics")
            with gr.Row():
                with gr.Column():
                    refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="primary")
                    

                    total_versions = gr.Number(
                        label="Total Content Versions",
                        value=0,
                        interactive=False
                    )
                    
                    versions_by_status = gr.JSON(
                        label="Versions by Status",
                        value={}
                    )
                
                with gr.Column():
                    avg_quality_score = gr.Number(
                        label="Average Quality Score",
                        value=0.0,
                        interactive=False
                    )
                    
                    recent_activity = gr.Textbox(
                        label="Recent Activity",
                        lines=8,
                        interactive=False
                    )
        

        scrape_btn.click(
            scrape_and_extract,
            inputs=[url_input],
            outputs=[scraped_content, screenshot_output]
        ).then(
            lambda: "✅ Content extracted successfully! Ready for AI processing.",
            outputs=[content_status]
        )
        
        send_to_ai_btn.click(
            lambda content: (content, "📤 Content sent to AI Processing tab"),
            inputs=[scraped_content],
            outputs=[pipeline_content, content_status]
        ).then(
            lambda content: content,
            inputs=[pipeline_content],
            outputs=[ai_input_display]
        )
        

        process_ai_btn.click(
            ai_process_content,
            inputs=[ai_input_display, ai_iterations],
            outputs=[ai_output_content]
        
        ).then(
            lambda: "✅ AI processing completed! Ready for human editing or finalization.",
            outputs=[ai_status]
        )
        
        send_to_editing_btn.click(
            lambda content: (content, "📤 Content sent to Human Editing tab"),
            inputs=[ai_output_content],
            outputs=[pipeline_content, ai_status]
        ).then(
            lambda content: (content, content),
            inputs=[pipeline_content],
            outputs=[human_input_display, human_edit_content]
        )
        
        skip_to_final_btn.click(
            lambda content: (content, "⏭️ Content sent directly to Final Processing tab"),
            inputs=[ai_output_content],
            outputs=[pipeline_content, ai_status]
        ).then(
            lambda content: content,
            inputs=[pipeline_content],
            outputs=[final_input_display]
        )
        

        save_edit_btn.click(
            save_human_edit,
            inputs=[human_edit_content, editor_notes],
            outputs=[edit_status, pipeline_content]
        )
        
        send_to_final_btn.click(
            lambda content: (content, "📤 Content sent to Final Processing tab"),
            inputs=[human_edit_content],
            outputs=[pipeline_content, edit_status]
        ).then(
            lambda content: content,
            inputs=[pipeline_content],
            outputs=[final_input_display]
        )
        

        finalize_btn.click(
            finalize_version,
            inputs=[final_input_display],
            outputs=[final_output_content]
        ).then(
            lambda: "🏁 Content finalized and ready for export!",
            outputs=[final_status]
        )
        
        export_btn.click(
            export_content,
            inputs=[final_output_content],
            outputs=[final_status]
        )
        
        restart_btn.click(
            lambda: ("", "", "", "", "", "", "🔄 Workflow reset. Ready to start new pipeline.", "", ""),
            outputs=[
                scraped_content, ai_input_display, ai_output_content, 
                human_input_display, human_edit_content, final_input_display, 
                final_output_content, content_status, pipeline_content
            ]
        ).then(
            lambda: ("Ready for extraction", "Waiting for content...", "Waiting for content...", "Waiting for content..."),
            outputs=[content_status, ai_status, edit_status, final_status]
        )
        

        search_btn.click(
            search_content,
            inputs=[search_query, status_filter],
            outputs=[search_results]
        )
        
        feedback_btn.click(
            provide_quality_feedback,
            inputs=[feedback_content_id, quality_rating],
            outputs=[feedback_status]
        )
        

        refresh_stats_btn.click(
            get_workflow_statistics,
            outputs=[total_versions, versions_by_status, avg_quality_score, recent_activity]
        )
        

        demo.load(
            get_workflow_statistics,
            outputs=[total_versions, versions_by_status, avg_quality_score, recent_activity]
        )
    
    return demo


def export_content_advanced(content, format_type="txt"):
    """Enhanced export function with multiple format support"""
    if not content.strip():
        return "No content to export"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        if format_type == "txt":
            filename = f"final_content_{timestamp}.txt"
            filepath = os.path.join(export_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif format_type == "md":
            filename = f"final_content_{timestamp}.md"
            filepath = os.path.join(export_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Book Content\n\n{content}")
                
        elif format_type == "json":
            filename = f"final_content_{timestamp}.json"
            filepath = os.path.join(export_dir, filename)
            export_data = {
                "timestamp": timestamp,
                "content": content,
                "word_count": len(content.split()),
                "export_format": format_type
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return f"✅ Content exported to: {filepath}"
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        return f"❌ Export failed: {str(e)}"


def batch_process_urls(urls_text, ai_iterations=2):
    """Process multiple URLs in batch"""
    if not urls_text.strip():
        return "No URLs provided"
    
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    results = []
    
    for i, url in enumerate(urls, 1):
        try:

            content, screenshot = workflow.scrape_content(url)
            if content:

                processed_content = workflow.ai_writing_pipeline(content, ai_iterations)
                results.append(f"✅ URL {i}: {url}\nProcessed successfully\n{'-'*50}\n")
            else:
                results.append(f"❌ URL {i}: {url}\nFailed to scrape\n{'-'*50}\n")
        except Exception as e:
            results.append(f"❌ URL {i}: {url}\nError: {str(e)}\n{'-'*50}\n")
    
    return "\n".join(results)


def calculate_advanced_quality_metrics(content):
    """Calculate advanced quality metrics for content"""
    if not content.strip():
        return {}
    
    words = content.split()
    sentences = [s.strip() for s in content.split('.') if s.strip()]
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    metrics = {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'avg_sentences_per_paragraph': len(sentences) / max(len(paragraphs), 1),
        'readability_score': min(len(words) / 1000, 1.0),  # Simplified readability
        'content_density': len(content.replace(' ', '')) / len(content) if content else 0
    }
    
    return metrics


class PerformanceMonitor:
    """Monitor workflow performance"""
    
    def __init__(self):
        self.metrics = {
            'scraping_times': [],
            'ai_processing_times': [],
            'total_requests': 0,
            'successful_operations': 0,
            'failed_operations': 0
        }
    
    def log_operation(self, operation_type, duration, success=True):
        """Log operation performance"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_operations'] += 1
            if operation_type == 'scraping':
                self.metrics['scraping_times'].append(duration)
            elif operation_type == 'ai_processing':
                self.metrics['ai_processing_times'].append(duration)
        else:
            self.metrics['failed_operations'] += 1
    
    def get_performance_report(self):
        """Get performance statistics"""
        if not self.metrics['total_requests']:
            return "No operations recorded yet"
        
        success_rate = (self.metrics['successful_operations'] / self.metrics['total_requests']) * 100
        
        report = f"""
**Performance Report**
- Total Operations: {self.metrics['total_requests']}
- Success Rate: {success_rate:.1f}%
- Failed Operations: {self.metrics['failed_operations']}

**Average Times:**
"""
        
        if self.metrics['scraping_times']:
            avg_scraping = sum(self.metrics['scraping_times']) / len(self.metrics['scraping_times'])
            report += f"- Scraping: {avg_scraping:.2f}s\n"
        
        if self.metrics['ai_processing_times']:
            avg_ai = sum(self.metrics['ai_processing_times']) / len(self.metrics['ai_processing_times'])
            report += f"- AI Processing: {avg_ai:.2f}s\n"
        
        return report


performance_monitor = PerformanceMonitor()


def main():
    """Main function to run the application"""
    try:

        demo = create_gradio_interface()
        

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()