"""
VeriGuard AI - Claim Extraction Engine
Advanced NLP-based claim extraction with entity recognition and classification.
"""

import os
import re
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple

from groq import Groq
from dotenv import load_dotenv

from core.models import ExtractedClaim, ClaimType
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache_result


load_dotenv()
logger = get_logger("extractor")


# Extraction prompts for different modes
EXTRACTION_PROMPTS = {
    "strict": """You are a precise fact extraction system. Extract ONLY explicit factual claims from the text.

Rules:
1. Extract ONLY statements that can be objectively verified
2. Ignore opinions, speculation, and subjective statements
3. Preserve numerical precision exactly as stated
4. Each claim should be self-contained and complete
5. Do NOT extract questions or hypotheticals
6. Maximum 10 claims per extraction

For each claim, classify its type:
- FACTUAL: General factual statements
- STATISTICAL: Claims involving numbers, percentages, or quantities
- TEMPORAL: Claims with time references or dates
- CAUSAL: Claims about cause-effect relationships
- COMPARATIVE: Claims comparing entities
- DEFINITIONAL: Definitions or classifications

Text to analyze:
{text}

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "text": "exact claim text",
      "type": "FACTUAL|STATISTICAL|TEMPORAL|CAUSAL|COMPARATIVE|DEFINITIONAL",
      "confidence": 0.0-1.0,
      "entities": ["entity1", "entity2"],
      "temporal_refs": ["date/time references if any"],
      "numbers": [{{"value": 123, "unit": "unit", "context": "what it measures"}}]
    }}
  ]
}}""",
    
    "standard": """You are a fact extraction system. Extract verifiable factual claims from the text.

Guidelines:
1. Extract statements that make factual assertions
2. Include statistical claims with numbers
3. Include temporal claims with dates/times
4. Include causal claims about relationships
5. Ignore pure opinions but include claims presented as facts
6. Preserve specificity and precision
7. Maximum 20 claims per extraction

For each claim, classify its type:
- FACTUAL: General factual statements
- STATISTICAL: Claims involving numbers, percentages, or quantities
- TEMPORAL: Claims with time references or dates
- CAUSAL: Claims about cause-effect relationships
- COMPARATIVE: Claims comparing entities
- DEFINITIONAL: Definitions or classifications

Text to analyze:
{text}

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "text": "exact claim text",
      "type": "FACTUAL|STATISTICAL|TEMPORAL|CAUSAL|COMPARATIVE|DEFINITIONAL",
      "confidence": 0.0-1.0,
      "entities": ["entity1", "entity2"],
      "temporal_refs": ["date/time references if any"],
      "numbers": [{{"value": 123, "unit": "unit", "context": "what it measures"}}]
    }}
  ]
}}""",
    
    "permissive": """You are a comprehensive fact extraction system. Extract all statements that could be verified.

Guidelines:
1. Extract any statement making an assertion about reality
2. Include implicit factual claims
3. Include claims embedded in questions or conditionals
4. Extract numerical data even without full context
5. Be inclusive - extract anything that could potentially be verified
6. Maximum 50 claims per extraction

For each claim, classify its type:
- FACTUAL: General factual statements
- STATISTICAL: Claims involving numbers, percentages, or quantities
- TEMPORAL: Claims with time references or dates
- CAUSAL: Claims about cause-effect relationships
- COMPARATIVE: Claims comparing entities
- DEFINITIONAL: Definitions or classifications

Text to analyze:
{text}

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "text": "exact claim text",
      "type": "FACTUAL|STATISTICAL|TEMPORAL|CAUSAL|COMPARATIVE|DEFINITIONAL",
      "confidence": 0.0-1.0,
      "entities": ["entity1", "entity2"],
      "temporal_refs": ["date/time references if any"],
      "numbers": [{{"value": 123, "unit": "unit", "context": "what it measures"}}]
    }}
  ]
}}"""
}


class ClaimExtractor:
    """
    Advanced claim extraction using LLM with:
    - Multiple extraction modes (strict, standard, permissive)
    - Claim type classification
    - Entity and temporal reference extraction
    - Numerical value parsing
    """
    
    def __init__(self, model: Optional[str] = None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY") or settings.llm.groq_api_key)
        self.model = model or settings.llm.model
        
        logger.info(f"ClaimExtractor initialized with model: {self.model}")
    
    def extract(
        self,
        text: str,
        mode: str = "standard",
        max_claims: int = 50
    ) -> List[ExtractedClaim]:
        """
        Extract factual claims from text.
        
        Args:
            text: Text to extract claims from
            mode: Extraction mode (strict, standard, permissive)
            max_claims: Maximum number of claims to extract
            
        Returns:
            List of ExtractedClaim objects
        """
        if not text or not text.strip():
            return []
        
        # Truncate very long texts
        if len(text) > 15000:
            logger.warning(f"Text truncated from {len(text)} to 15000 chars")
            text = text[:15000]
        
        # Get appropriate prompt
        prompt_template = EXTRACTION_PROMPTS.get(mode, EXTRACTION_PROMPTS["standard"])
        prompt = prompt_template.format(text=text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise fact extraction system. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            claims = self._parse_response(content, max_claims)
            
            logger.info(f"Extracted {len(claims)} claims from text ({len(text)} chars)")
            return claims
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Fallback to simple extraction
            return self._fallback_extract(text, max_claims)
    
    def _parse_response(self, content: str, max_claims: int) -> List[ExtractedClaim]:
        """Parse LLM response into ExtractedClaim objects."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            # Try array format
            json_match = re.search(r'\[[\s\S]*\]', content)
        
        if not json_match:
            logger.warning("No JSON found in response")
            return []
        
        try:
            data = json.loads(json_match.group())
            
            # Handle both {"claims": [...]} and [...] formats
            if isinstance(data, dict):
                claims_data = data.get("claims", [])
            else:
                claims_data = data
            
            claims = []
            for i, item in enumerate(claims_data[:max_claims]):
                claim_text = item.get("text", "") if isinstance(item, dict) else str(item)
                
                if not claim_text:
                    continue
                
                # Parse claim type
                claim_type_str = item.get("type", "FACTUAL") if isinstance(item, dict) else "FACTUAL"
                try:
                    claim_type = ClaimType(claim_type_str.upper())
                except ValueError:
                    claim_type = ClaimType.FACTUAL
                
                claim = ExtractedClaim(
                    id=str(uuid.uuid4())[:8],
                    text=claim_text,
                    claim_type=claim_type,
                    confidence=float(item.get("confidence", 0.8)) if isinstance(item, dict) else 0.8,
                    entities=item.get("entities", []) if isinstance(item, dict) else [],
                    temporal_references=item.get("temporal_refs", []) if isinstance(item, dict) else [],
                    numerical_values=item.get("numbers", []) if isinstance(item, dict) else []
                )
                claims.append(claim)
            
            return claims
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return []
    
    def _fallback_extract(self, text: str, max_claims: int) -> List[ExtractedClaim]:
        """Simple rule-based extraction as fallback."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences[:max_claims]:
            sentence = sentence.strip()
            
            # Skip very short or very long sentences
            if len(sentence) < 10 or len(sentence) > 500:
                continue
            
            # Skip questions
            if sentence.endswith('?'):
                continue
            
            # Basic heuristics for factual content
            factual_indicators = [
                r'\b\d+\b',  # Contains numbers
                r'\b(is|are|was|were|has|have|had)\b',  # Copula verbs
                r'\b(located|founded|discovered|invented|created)\b',  # Factual verbs
            ]
            
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators):
                claims.append(ExtractedClaim(
                    id=str(uuid.uuid4())[:8],
                    text=sentence,
                    claim_type=ClaimType.FACTUAL,
                    confidence=0.5  # Lower confidence for fallback
                ))
        
        return claims
    
    def extract_batch(
        self,
        texts: List[str],
        mode: str = "standard",
        max_claims_per_text: int = 20
    ) -> List[List[ExtractedClaim]]:
        """
        Extract claims from multiple texts.
        
        Args:
            texts: List of texts to process
            mode: Extraction mode
            max_claims_per_text: Maximum claims per text
            
        Returns:
            List of claim lists, one per input text
        """
        return [
            self.extract(text, mode, max_claims_per_text)
            for text in texts
        ]


# Global extractor instance
_extractor: Optional[ClaimExtractor] = None


def get_extractor() -> ClaimExtractor:
    """Get or create the extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor()
    return _extractor
