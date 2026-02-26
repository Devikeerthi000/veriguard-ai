from app.embedder import Embedder
from app.index_builder import IndexBuilder
from app.retriever import Retriever
from app.claim_extractor import ClaimExtractor
from app.verifier import Verifier
from app.risk_engine import RiskEngine

def load_knowledge_base(path="data/knowledge_base.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    print("🚀 VeriGuard AI - Hallucination Detection Engine\n")

    # Load knowledge base
    knowledge_docs = load_knowledge_base()

    # Build embedding index
    embedder = Embedder()
    embeddings = embedder.embed(knowledge_docs)

    dimension = embeddings.shape[1]
    index_builder = IndexBuilder(dimension)
    index_builder.build(embeddings, knowledge_docs)

    retriever = Retriever(embedder, index_builder)
    claim_extractor = ClaimExtractor()
    verifier = Verifier()
    risk_engine = RiskEngine()

    text = input("Enter LLM Output Text:\n")

    claims = claim_extractor.extract(text)

    if not claims:
        print("No factual claims detected.")
        return

    for claim in claims:
        print(f"\n🔎 Claim: {claim}")

        evidence = retriever.retrieve(claim)
        print("Retrieved Evidence:", evidence)

        verification = verifier.verify(claim, evidence)
        risk = risk_engine.calculate(verification)

        print("Status:", verification["status"])
        print("Confidence:", verification["confidence"])
        print("Severity:", risk["severity"])
        print("Risk Score:", risk["risk_score"])
        print("Explanation:", verification["explanation"])

if __name__ == "__main__":
    main()