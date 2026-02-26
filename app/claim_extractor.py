import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ClaimExtractor:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def extract(self, text):
        prompt = f"""
Extract factual claims from the following text.
Ignore opinions.
Return ONLY a valid JSON list of strings.

Text:
{text}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        print("Raw LLM Response:", content)

        try:
            return json.loads(content)
        except:
            return []