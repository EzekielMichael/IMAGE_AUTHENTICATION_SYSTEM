# authenticate/llm_qa.py

import os
import json
import re
import logging
from typing import Tuple

from django.conf import settings
from .models import UploadedImage, AIDetection, HumanEditDetection, HeatmapAnalysis, FinalVerdict, MetadataAnalysis, QnAHistory

logger = logging.getLogger(__name__)

class LLMQA:
    def __init__(self, provider=None):
        # Determine provider: argument > settings > default
        self.provider = provider or getattr(settings, 'LLM_PROVIDER', 'gemini')
        self.provider = self.provider.lower()
        
        # Load provider-specific config
        if self.provider == "gemini":
            # First try settings, then environment
            self.api_key = getattr(settings, 'GEMINI_API_KEY', None) or os.environ.get('GEMINI_API_KEY')
            self.model_name = getattr(settings, 'GEMINI_MODEL', None) or os.environ.get('GEMINI_MODEL', 'gemini-3.6-flash')
            
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set in settings or environment.")
            
            # Use the new google.genai client
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized with model: {self.model_name}")
            except ImportError:
                raise ImportError("Please install google-genai: pip install google-genai")
                
        elif self.provider in ["deepseek", "groq", "openai"]:
            self.api_key = getattr(settings, 'LLM_API_KEY', None) or os.environ.get('LLM_API_KEY')
            self.api_url = getattr(settings, 'LLM_API_URL', None) or os.environ.get('LLM_API_URL')
            self.model_name = getattr(settings, 'LLM_MODEL', None) or os.environ.get('LLM_MODEL', 'deepseek-v4-pro')
            if not self.api_key:
                raise ValueError(f"LLM_API_KEY is not set for provider: {self.provider}")
                
        elif self.provider == "ollama":
            self.api_url = getattr(settings, 'LLM_API_URL', None) or os.environ.get('LLM_API_URL', 'http://localhost:11434')
            self.model_name = getattr(settings, 'LLM_MODEL', None) or os.environ.get('LLM_MODEL', 'llama3')
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _build_system_prompt(self, image_data: dict) -> str:
        """Build the system prompt from image analysis data."""
        verdict = image_data.get('verdict', 'N/A')
        ai_label = image_data.get('ai_label', 'N/A')
        ai_conf = image_data.get('ai_confidence', 0)
        human_score = image_data.get('human_score', 0)
        overall_conf = image_data.get('overall_confidence', 0)
        metadata = image_data.get('metadata', {})
        heatmap = image_data.get('heatmap', {})

        meta_lines = []
        if metadata.get('camera_make'):
            meta_lines.append(f"Camera: {metadata['camera_make']} {metadata['camera_model']}")
        if metadata.get('date_original'):
            meta_lines.append(f"Capture Date: {metadata['date_original']}")
        if metadata.get('software'):
            meta_lines.append(f"Software: {metadata['software']}")
        if metadata.get('ai_tool'):
            meta_lines.append(f"AI Tool Identified: {metadata['ai_tool']}")
        meta_lines.append(f"C2PA Digital Signatures: {'Present' if metadata.get('c2pa') else 'Not present'}")
        if metadata.get('gps_lat') and metadata.get('gps_lon'):
            meta_lines.append(f"GPS: {metadata['gps_lat']}, {metadata['gps_lon']}")

        heatmap_lines = []
        if heatmap.get('primary_region'):
            heatmap_lines.append(f"Primary Edited Region: {heatmap['primary_region']}")
        if heatmap.get('secondary_region'):
            heatmap_lines.append(f"Secondary Edited Region: {heatmap['secondary_region']}")
        if heatmap.get('edited_percent') is not None:
            heatmap_lines.append(f"Edited Percentage: {heatmap['edited_percent']:.1f}%")
        if heatmap.get('severity'):
            heatmap_lines.append(f"Severity Level: {heatmap['severity']}")
        if heatmap.get('zones'):
            heatmap_lines.append(f"Number of Edit Zones: {heatmap['zones']}")

        prompt = f"""
            You are Forensix AI, a forensic image analysis assistant. You have access to the following analysis 
            results for the image being discussed. You must answer the user's question using ONLY the information provided below.
            Do NOT invent any facts or add external knowledge. If the question cannot be answered from the given data, politely say so.

=== ANALYSIS RESULTS ===
Final Verdict: {verdict}
Overall Confidence: {overall_conf:.1f}%
AI Detection: {ai_label} (Confidence: {ai_conf:.1f}%)
Human Editing Score: {human_score:.1f}% (Threshold for editing: 35%)

--- Metadata ---
{chr(10).join(meta_lines) if meta_lines else "No metadata available"}

--- Heatmap / Editing Localization ---
{chr(10).join(heatmap_lines) if heatmap_lines else "No heatmap data available"}

Additional Notes:
- Red areas in heatmap indicate strong editing probability.
- C2PA signatures help verify authenticity but are also added by AI tools.
- The AI model used is ResNet50.

Answer the user's question in a clear, conversational tone. Be helpful but concise. Keep responses under 200 words.
"""
        return prompt

    def _get_gemini_response(self, system_prompt: str, question: str) -> str:
        """Call Gemini API using the new google.genai client."""
        full_prompt = f"{system_prompt}\n\nUser Question: {question}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Check for specific errors
            error_msg = str(e)
            if "quota" in error_msg.lower():
                raise Exception("Gemini API quota exceeded. Please try again later.")
            elif "key" in error_msg.lower() or "auth" in error_msg.lower():
                raise Exception("Invalid Gemini API key. Please check your configuration.")
            else:
                raise e

    def _get_openai_compatible_response(self, system_prompt: str, question: str) -> str:
        """Call OpenAI-compatible API (DeepSeek, Groq, etc.)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
            
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()

    def _get_ollama_response(self, system_prompt: str, question: str) -> str:
        """Call Ollama local API."""
        try:
            import requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
            
        full_prompt = f"{system_prompt}\n\nUser Question: {question}"
        response = requests.post(
            f"{self.api_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False
            }
        )
        data = response.json()
        return data.get("response", "").strip()

    def get_answer(self, image_id: int, question: str) -> Tuple[str, bool]:
        """Get answer from LLM, with caching."""
        # Check cache
        cached = QnAHistory.objects.filter(image_id=image_id, question__iexact=question).first()
        if cached:
            return cached.answer, True

        # Fetch data
        try:
            image = UploadedImage.objects.get(id=image_id)
            ai = AIDetection.objects.filter(image=image).first()
            human = HumanEditDetection.objects.filter(image=image).first()
            heatmap = HeatmapAnalysis.objects.filter(image=image).first()
            verdict = FinalVerdict.objects.filter(image=image).first()
            metadata = MetadataAnalysis.objects.filter(image=image).first()
        except UploadedImage.DoesNotExist:
            return "Image not found in database.", False

        data = {
            'verdict': verdict.verdict if verdict else "Unknown",
            'overall_confidence': verdict.confidence_score if verdict else 0,
            'ai_label': ai.label if ai else "Unknown",
            'ai_confidence': ai.confidence if ai else 0,
            'human_score': human.detection_score * 100 if human else 0,
            'metadata': {
                'camera_make': metadata.camera_make if metadata else None,
                'camera_model': metadata.camera_model if metadata else None,
                'date_original': metadata.date_original if metadata else None,
                'software': metadata.software if metadata else None,
                'ai_tool': metadata.ai_tool if metadata else None,
                'c2pa': metadata.c2pa_present if metadata else False,
                'gps_lat': metadata.gps_latitude if metadata else None,
                'gps_lon': metadata.gps_longitude if metadata else None,
            },
            'heatmap': {
                'primary_region': heatmap.primary_region if heatmap else None,
                'secondary_region': heatmap.secondary_region if heatmap else None,
                'edited_percent': heatmap.edited_percentage if heatmap else None,
                'severity': heatmap.severity_level if heatmap else None,
                'zones': heatmap.edit_zones_count if heatmap else None,
            }
        }

        system_prompt = self._build_system_prompt(data)

        try:
            if self.provider == "gemini":
                answer = self._get_gemini_response(system_prompt, question)
            elif self.provider in ["deepseek", "groq", "openai"]:
                answer = self._get_openai_compatible_response(system_prompt, question)
            elif self.provider == "ollama":
                answer = self._get_ollama_response(system_prompt, question)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Cache the answer
            QnAHistory.objects.create(
                image_id=image_id,
                question=question,
                answer=answer,
                question_type="LLM",
                used_ai_confidence=ai.confidence if ai else 0,
                used_human_score=human.detection_score if human else 0,
                used_primary_region=heatmap.primary_region if heatmap else "",
                used_verdict=verdict.verdict if verdict else "",
                was_from_cache=False
            )
            return answer, False

        except Exception as e:
            logger.error(f"LLM QA error: {e}", exc_info=True)
            # Fallback to rule-based QA
            from .qa_system import qa_system
            answer, _ = qa_system.get_answer(image_id, question)
            return answer, False