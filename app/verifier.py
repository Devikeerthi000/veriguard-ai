import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Verifier:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def verify(self, claim, evidence):
        evidence_text = "\n".join(evidence)

        prompt = f"""
You are a factual verification system.

Claim:
"{claim}"

Evidence:
{evidence_text}

Classify the claim as:
Supported
Contradicted
Unverifiable

Return ONLY valid JSON:
{{
    "status": "",
    "confidence": 0-1,
    "explanation": ""
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except:
            return {
                "status": "Unverifiable",
                "confidence": 0.5,
                "explanation": "LLM response parsing failed."
            }