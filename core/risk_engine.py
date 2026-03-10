"""
VeriGuard AI - Risk Assessment Engine
Multi-dimensional risk scoring with impact analysis and recommendations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.models import (
    VerificationResult, RiskAssessment, VerificationStatus, 
    SeverityLevel, ClaimType, ExtractedClaim
)
from config.settings import settings
from utils.logger import get_logger


logger = get_logger("risk_engine")


# Risk factor weights
RISK_WEIGHTS = {
    # Status-based risk
    "status": {
        VerificationStatus.SUPPORTED: 0.0,
        VerificationStatus.PARTIALLY_SUPPORTED: 0.3,
        VerificationStatus.UNVERIFIABLE: 0.5,
        VerificationStatus.INSUFFICIENT_EVIDENCE: 0.4,
        VerificationStatus.CONTRADICTED: 1.0,
    },
    
    # Claim type sensitivity (some claims are higher risk if wrong)
    "claim_type": {
        ClaimType.STATISTICAL: 0.8,   # Numbers need to be precise
        ClaimType.TEMPORAL: 0.7,      # Dates/times matter
        ClaimType.CAUSAL: 0.6,        # Cause-effect claims
        ClaimType.FACTUAL: 0.5,       # General facts
        ClaimType.COMPARATIVE: 0.4,   # Comparisons
        ClaimType.DEFINITIONAL: 0.3,  # Definitions
    },
    
    # Domain sensitivity (some domains are higher risk)
    "domain": {
        "medicine": 0.95,
        "medical": 0.95,
        "health": 0.9,
        "legal": 0.85,
        "financial": 0.8,
        "scientific": 0.7,
        "historical": 0.5,
        "general": 0.4,
    }
}

# Impact categories based on domain
IMPACT_CATEGORIES = {
    "medical": "Health & Safety",
    "medicine": "Health & Safety",
    "health": "Health & Safety",
    "legal": "Legal & Compliance",
    "financial": "Financial & Economic",
    "scientific": "Scientific Accuracy",
    "historical": "Historical Record",
    "geopolitical": "Public Information",
    "general": "General Information"
}

# Recommended actions by severity
RECOMMENDATIONS = {
    SeverityLevel.CRITICAL: "URGENT: Do not publish. Requires immediate fact-check by domain expert.",
    SeverityLevel.HIGH: "CAUTION: Significant accuracy concerns. Manual verification required before use.",
    SeverityLevel.MEDIUM: "ADVISORY: Some uncertainty exists. Consider adding qualifiers or sources.",
    SeverityLevel.LOW: "ACCEPTABLE: Minor concerns. Review for precision but generally reliable.",
    SeverityLevel.NEGLIGIBLE: "VERIFIED: Claim appears accurate based on available evidence."
}


class RiskEngine:
    """
    Multi-dimensional risk assessment engine with:
    - Status-based risk scoring
    - Claim type sensitivity weighting
    - Domain-aware impact assessment
    - Confidence calibration
    - Actionable recommendations
    """
    
    def __init__(self):
        self.confidence_threshold = settings.verification.confidence_threshold
        logger.info("RiskEngine initialized")
    
    def assess(
        self,
        claim: ExtractedClaim,
        verification: VerificationResult
    ) -> RiskAssessment:
        """
        Assess risk for a verified claim.
        
        Args:
            claim: The original claim
            verification: Verification result
            
        Returns:
            RiskAssessment with severity, score, and recommendations
        """
        # Calculate base risk from verification status
        base_risk = RISK_WEIGHTS["status"].get(
            verification.status, 
            0.5
        )
        
        # Adjust for confidence (low confidence = higher risk)
        confidence_factor = 1.0 - (verification.confidence * 0.3)
        
        # Adjust for claim type sensitivity
        type_factor = RISK_WEIGHTS["claim_type"].get(
            claim.claim_type,
            0.5
        )
        
        # Determine domain from evidence categories
        domain = self._determine_domain(verification)
        domain_factor = RISK_WEIGHTS["domain"].get(domain, 0.5)
        
        # Additional risk factors
        risk_factors = []
        additional_risk = 0.0
        
        # Numerical discrepancy risk
        if verification.numerical_accuracy:
            if verification.numerical_accuracy.get("has_discrepancy", False):
                additional_risk += 0.15
                risk_factors.append("Numerical discrepancy detected")
        
        # Temporal inconsistency risk
        if verification.temporal_validity:
            if verification.temporal_validity.get("has_inconsistency", False):
                additional_risk += 0.1
                risk_factors.append("Temporal inconsistency detected")
        
        # Evidence contradiction risk
        if verification.contradictions_found:
            additional_risk += 0.1 * len(verification.contradictions_found)
            risk_factors.append(f"{len(verification.contradictions_found)} contradiction(s) in evidence")
        
        # Insufficient evidence risk
        if len(verification.evidence_used) < 2:
            additional_risk += 0.1
            risk_factors.append("Limited evidence available")
        
        # Calculate final risk score
        risk_score = (
            base_risk * 0.4 +           # Status weight
            confidence_factor * 0.2 +    # Confidence weight
            type_factor * 0.15 +         # Claim type weight
            domain_factor * 0.15 +       # Domain weight
            additional_risk * 0.1        # Additional factors
        )
        
        # Clamp to [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Determine severity level
        severity = self._calculate_severity(risk_score, verification.status)
        
        # Get impact category
        impact_category = IMPACT_CATEGORIES.get(domain, "General Information")
        
        # Get recommendation
        recommendation = RECOMMENDATIONS.get(severity, RECOMMENDATIONS[SeverityLevel.MEDIUM])
        
        return RiskAssessment(
            claim_id=claim.id,
            severity=severity,
            risk_score=round(risk_score, 3),
            impact_category=impact_category,
            recommended_action=recommendation,
            factors=risk_factors
        )
    
    def _determine_domain(self, verification: VerificationResult) -> str:
        """Determine the domain based on evidence categories."""
        if not verification.evidence_used:
            return "general"
        
        # Count evidence categories
        categories = {}
        for ev in verification.evidence_used:
            cat = ev.source_category.lower()
            categories[cat] = categories.get(cat, 0) + 1
        
        # Map categories to domains
        domain_mapping = {
            "medicine health": "medical",
            "medicine_health": "medical",
            "medical": "medical",
            "health": "health",
            "science technology": "scientific",
            "science_technology": "scientific",
            "scientific": "scientific",
            "business finance": "financial",
            "business_finance": "financial",
            "financial": "financial",
            "legal": "legal",
            "history culture": "historical",
            "history_culture": "historical",
            "historical": "historical",
            "geopolitical": "geopolitical"
        }
        
        # Find most common mapped domain
        for cat in categories:
            for key, domain in domain_mapping.items():
                if key in cat:
                    return domain
        
        return "general"
    
    def _calculate_severity(
        self,
        risk_score: float,
        status: VerificationStatus
    ) -> SeverityLevel:
        """Calculate severity level from risk score and status."""
        
        # Override for contradicted claims
        if status == VerificationStatus.CONTRADICTED:
            if risk_score > 0.7:
                return SeverityLevel.CRITICAL
            return SeverityLevel.HIGH
        
        # Score-based severity
        if risk_score >= 0.8:
            return SeverityLevel.CRITICAL
        elif risk_score >= 0.6:
            return SeverityLevel.HIGH
        elif risk_score >= 0.4:
            return SeverityLevel.MEDIUM
        elif risk_score >= 0.2:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NEGLIGIBLE
    
    def assess_batch(
        self,
        claims: List[ExtractedClaim],
        verifications: List[VerificationResult]
    ) -> List[RiskAssessment]:
        """
        Assess risk for multiple claims.
        
        Args:
            claims: List of claims
            verifications: Corresponding verification results
            
        Returns:
            List of RiskAssessments
        """
        return [
            self.assess(claim, verification)
            for claim, verification in zip(claims, verifications)
        ]
    
    def calculate_aggregate_risk(
        self,
        assessments: List[RiskAssessment]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate risk metrics for a set of assessments.
        
        Args:
            assessments: List of risk assessments
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not assessments:
            return {
                "overall_score": 0.0,
                "overall_severity": SeverityLevel.NEGLIGIBLE,
                "hallucination_rate": 0.0,
                "severity_distribution": {},
                "high_risk_count": 0
            }
        
        # Calculate metrics
        scores = [a.risk_score for a in assessments]
        severities = [a.severity for a in assessments]
        
        # Overall score (weighted by severity)
        severity_weights = {
            SeverityLevel.CRITICAL: 2.0,
            SeverityLevel.HIGH: 1.5,
            SeverityLevel.MEDIUM: 1.0,
            SeverityLevel.LOW: 0.5,
            SeverityLevel.NEGLIGIBLE: 0.25
        }
        
        weighted_scores = [
            score * severity_weights.get(sev, 1.0)
            for score, sev in zip(scores, severities)
        ]
        overall_score = sum(weighted_scores) / sum(severity_weights.get(s, 1.0) for s in severities)
        overall_score = min(1.0, overall_score)
        
        # Severity distribution
        severity_dist = {}
        for sev in SeverityLevel:
            count = sum(1 for s in severities if s == sev)
            severity_dist[sev.value] = count
        
        # High risk count
        high_risk = sum(1 for s in severities if s in [SeverityLevel.CRITICAL, SeverityLevel.HIGH])
        
        # Hallucination rate (contradicted claims)
        # This requires access to verification results - approximate from severity
        estimated_hallucination = high_risk / len(assessments) if assessments else 0
        
        # Overall severity
        if any(s == SeverityLevel.CRITICAL for s in severities):
            overall_severity = SeverityLevel.CRITICAL
        elif high_risk > len(assessments) * 0.3:
            overall_severity = SeverityLevel.HIGH
        elif overall_score > 0.5:
            overall_severity = SeverityLevel.MEDIUM
        elif overall_score > 0.3:
            overall_severity = SeverityLevel.LOW
        else:
            overall_severity = SeverityLevel.NEGLIGIBLE
        
        return {
            "overall_score": round(overall_score, 3),
            "overall_severity": overall_severity,
            "hallucination_rate": round(estimated_hallucination, 3),
            "severity_distribution": severity_dist,
            "high_risk_count": high_risk,
            "total_claims": len(assessments)
        }


# Global risk engine instance
_risk_engine: Optional[RiskEngine] = None


def get_risk_engine() -> RiskEngine:
    """Get or create the risk engine singleton."""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine()
    return _risk_engine
