"""
VeriGuard AI - Multi-Stage Verification Engine
Advanced verification with contradiction detection, temporal analysis, and numeric precision.
"""

import os
import re
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

from core.models import (
    ExtractedClaim, Evidence, VerificationResult, 
    VerificationStatus, ClaimType
)
from config.settings import settings
from utils.logger import get_logger


load_dotenv()
logger = get_logger("verifier")


VERIFICATION_PROMPT = """You are an expert fact verification system. Verify the claim against the provided evidence.

CLAIM:
"{claim}"

CLAIM TYPE: {claim_type}

EVIDENCE:
{evidence}

Analyze carefully and determine:
1. Does the evidence SUPPORT, CONTRADICT, or leave the claim UNVERIFIABLE?
2. Are there any NUMERICAL discrepancies?
3. Are there any TEMPORAL inconsistencies (dates, timeframes)?
4. What is the strength of the evidence?

Classification guidelines:
- SUPPORTED: Evidence directly confirms the claim
- CONTRADICTED: Evidence directly contradicts the claim
- PARTIALLY_SUPPORTED: Some aspects supported, others not
- UNVERIFIABLE: Evidence is relevant but insufficient
- INSUFFICIENT_EVIDENCE: No relevant evidence found

Return ONLY valid JSON:
{{
    "status": "SUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED|UNVERIFIABLE|INSUFFICIENT_EVIDENCE",
    "confidence": 0.0-1.0,
    "explanation": "detailed reasoning",
    "supporting_facts": ["list of facts that support the claim"],
    "contradictions": ["list of facts that contradict the claim"],
    "numerical_analysis": {{
        "has_discrepancy": true/false,
        "claimed_value": "value from claim",
        "evidence_value": "value from evidence",
        "discrepancy_type": "exact|approximate|order_of_magnitude|none"
    }},
    "temporal_analysis": {{
        "has_inconsistency": true/false,
        "claimed_time": "time reference from claim",
        "evidence_time": "time reference from evidence",
        "inconsistency_type": "date|duration|sequence|none"
    }}
}}"""


CONTRADICTION_DETECTION_PROMPT = """Analyze whether these two statements contradict each other.

Statement 1: "{stmt1}"
Statement 2: "{stmt2}"

Consider:
1. Direct contradictions (opposite claims)
2. Numerical inconsistencies
3. Temporal conflicts
4. Logical incompatibilities

Return JSON:
{{
    "contradicts": true/false,
    "type": "direct|numerical|temporal|logical|none",
    "explanation": "brief explanation"
}}"""


class VerificationEngine:
    """
    Advanced verification engine with:
    - Multi-evidence synthesis
    - Contradiction detection
    - Temporal consistency checking
    - Numerical precision analysis
    - Confidence calibration
    """
    
    def __init__(self, model: Optional[str] = None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY") or settings.llm.groq_api_key)
        self.model = model or settings.llm.model
        
        # Feature flags from config
        self.enable_contradiction = settings.verification.enable_contradiction_detection
        self.enable_temporal = settings.verification.enable_temporal_analysis
        self.enable_numeric = settings.verification.enable_numeric_precision
        
        logger.info(f"VerificationEngine initialized with model: {self.model}")
    
    def verify(
        self,
        claim: ExtractedClaim,
        evidence: List[Evidence],
        depth: str = "standard"
    ) -> VerificationResult:
        """
        Verify a claim against retrieved evidence.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence documents
            depth: Verification depth (quick, standard, thorough)
            
        Returns:
            VerificationResult with status, confidence, and analysis
        """
        if not evidence:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.1,
                explanation="No relevant evidence found in knowledge base.",
                evidence_used=[]
            )
        
        # Format evidence for prompt
        evidence_text = self._format_evidence(evidence)
        
        # Build verification prompt
        prompt = VERIFICATION_PROMPT.format(
            claim=claim.text,
            claim_type=claim.claim_type.value,
            evidence=evidence_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fact verification system. Analyze claims against evidence precisely and respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content.strip()
            result = self._parse_verification_response(content, claim, evidence)
            
            # Additional analysis for thorough mode
            if depth == "thorough":
                result = self._enhanced_analysis(claim, evidence, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.3,
                explanation=f"Verification error: {str(e)}",
                evidence_used=evidence
            )
    
    def _format_evidence(self, evidence: List[Evidence]) -> str:
        """Format evidence list for prompt."""
        formatted = []
        for i, ev in enumerate(evidence, 1):
            formatted.append(f"[{i}] (Score: {ev.relevance_score:.2f}, Category: {ev.source_category})")
            formatted.append(f"    {ev.text}")
        return "\n".join(formatted)
    
    def _parse_verification_response(
        self,
        content: str,
        claim: ExtractedClaim,
        evidence: List[Evidence]
    ) -> VerificationResult:
        """Parse LLM verification response."""
        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.3,
                explanation="Could not parse verification response.",
                evidence_used=evidence
            )
        
        try:
            data = json.loads(json_match.group())
            
            # Parse status
            status_str = data.get("status", "UNVERIFIABLE").upper()
            try:
                status = VerificationStatus(status_str)
            except ValueError:
                status = VerificationStatus.UNVERIFIABLE
            
            # Build result
            result = VerificationResult(
                claim_id=claim.id,
                status=status,
                confidence=float(data.get("confidence", 0.5)),
                explanation=data.get("explanation", "No explanation provided."),
                evidence_used=evidence,
                supporting_facts=data.get("supporting_facts", []),
                contradictions_found=data.get("contradictions", [])
            )
            
            # Add numerical analysis
            if "numerical_analysis" in data:
                result.numerical_accuracy = data["numerical_analysis"]
            
            # Add temporal analysis
            if "temporal_analysis" in data:
                result.temporal_validity = data["temporal_analysis"]
            
            return result
            
        except json.JSONDecodeError:
            return VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.3,
                explanation="JSON parsing error in verification.",
                evidence_used=evidence
            )
    
    def _enhanced_analysis(
        self,
        claim: ExtractedClaim,
        evidence: List[Evidence],
        result: VerificationResult
    ) -> VerificationResult:
        """Perform enhanced analysis for thorough verification."""
        
        # Check for internal contradictions in evidence
        if self.enable_contradiction and len(evidence) > 1:
            contradictions = self._detect_evidence_contradictions(evidence)
            if contradictions:
                result.contradictions_found.extend(contradictions)
                # Adjust confidence if evidence is contradictory
                result.confidence = min(result.confidence, 0.6)
        
        # Numerical precision check
        if self.enable_numeric and claim.numerical_values:
            numeric_check = self._check_numerical_precision(claim, evidence)
            if numeric_check:
                result.numerical_accuracy = result.numerical_accuracy or {}
                result.numerical_accuracy.update(numeric_check)
        
        # Temporal consistency check
        if self.enable_temporal and claim.temporal_references:
            temporal_check = self._check_temporal_consistency(claim, evidence)
            if temporal_check:
                result.temporal_validity = result.temporal_validity or {}
                result.temporal_validity.update(temporal_check)
        
        return result
    
    def _detect_evidence_contradictions(
        self,
        evidence: List[Evidence]
    ) -> List[str]:
        """Detect contradictions between evidence documents."""
        contradictions = []
        
        for i, ev1 in enumerate(evidence):
            for ev2 in evidence[i+1:]:
                try:
                    prompt = CONTRADICTION_DETECTION_PROMPT.format(
                        stmt1=ev1.text,
                        stmt2=ev2.text
                    )
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=256
                    )
                    
                    content = response.choices[0].message.content.strip()
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    
                    if json_match:
                        data = json.loads(json_match.group())
                        if data.get("contradicts", False):
                            contradictions.append(
                                f"Evidence conflict: {data.get('explanation', 'No details')}"
                            )
                            
                except Exception as e:
                    logger.warning(f"Contradiction check failed: {e}")
        
        return contradictions
    
    def _check_numerical_precision(
        self,
        claim: ExtractedClaim,
        evidence: List[Evidence]
    ) -> Optional[Dict[str, Any]]:
        """Check numerical values against evidence."""
        # Extract numbers from claim
        claim_numbers = re.findall(
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|million|billion|trillion)?',
            claim.text
        )
        
        if not claim_numbers:
            return None
        
        # Extract numbers from evidence
        evidence_numbers = []
        for ev in evidence:
            numbers = re.findall(
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|million|billion|trillion)?',
                ev.text
            )
            evidence_numbers.extend(numbers)
        
        # Simple comparison (could be enhanced with unit conversion)
        discrepancies = []
        for claim_num, claim_unit in claim_numbers:
            claim_val = float(claim_num.replace(",", ""))
            
            for ev_num, ev_unit in evidence_numbers:
                ev_val = float(ev_num.replace(",", ""))
                
                # Check for significant discrepancy
                if claim_val > 0 and ev_val > 0:
                    ratio = max(claim_val, ev_val) / min(claim_val, ev_val)
                    if ratio > 1.1:  # More than 10% difference
                        discrepancies.append({
                            "claimed": f"{claim_num} {claim_unit or ''}".strip(),
                            "evidence": f"{ev_num} {ev_unit or ''}".strip(),
                            "ratio": ratio
                        })
        
        if discrepancies:
            return {
                "has_discrepancy": True,
                "discrepancies": discrepancies
            }
        
        return {"has_discrepancy": False}
    
    def _check_temporal_consistency(
        self,
        claim: ExtractedClaim,
        evidence: List[Evidence]
    ) -> Optional[Dict[str, Any]]:
        """Check temporal references for consistency."""
        # Extract years from claim and evidence
        claim_years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', claim.text)
        
        evidence_years = []
        for ev in evidence:
            years = re.findall(r'\b(1[89]\d{2}|20\d{2})\b', ev.text)
            evidence_years.extend(years)
        
        if not claim_years or not evidence_years:
            return None
        
        # Check for year mismatches
        mismatches = []
        for cy in claim_years:
            if cy not in evidence_years:
                # Check if similar events have different years in evidence
                for ey in evidence_years:
                    if abs(int(cy) - int(ey)) > 0 and abs(int(cy) - int(ey)) < 50:
                        mismatches.append({
                            "claimed_year": cy,
                            "evidence_year": ey
                        })
        
        if mismatches:
            return {
                "has_inconsistency": True,
                "mismatches": mismatches
            }
        
        return {"has_inconsistency": False}
    
    def verify_batch(
        self,
        claims: List[ExtractedClaim],
        evidence_map: Dict[str, List[Evidence]],
        depth: str = "standard"
    ) -> List[VerificationResult]:
        """
        Verify multiple claims.
        
        Args:
            claims: List of claims to verify
            evidence_map: Mapping of claim text to evidence
            depth: Verification depth
            
        Returns:
            List of VerificationResults
        """
        results = []
        for claim in claims:
            evidence = evidence_map.get(claim.text, [])
            result = self.verify(claim, evidence, depth)
            results.append(result)
        return results


# Global verifier instance
_verifier: Optional[VerificationEngine] = None


def get_verifier() -> VerificationEngine:
    """Get or create the verifier singleton."""
    global _verifier
    if _verifier is None:
        _verifier = VerificationEngine()
    return _verifier
