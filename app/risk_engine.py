class RiskEngine:
    def calculate(self, verification_result):
        status = verification_result.get("status", "Unverifiable")
        confidence = verification_result.get("confidence", 0.5)

        if status == "Contradicted":
            severity = "High"
            risk_score = round(1 - confidence, 2)
        elif status == "Unverifiable":
            severity = "Medium"
            risk_score = 0.6
        else:
            severity = "Low"
            risk_score = 0.2

        return {
            "severity": severity,
            "risk_score": risk_score
        }