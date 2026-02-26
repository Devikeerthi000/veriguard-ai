import faiss
import numpy as np

class IndexBuilder:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def build(self, embeddings, documents):
        self.documents = documents
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, top_k=3):
        query_embedding = query_embedding.astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.documents[i] for i in indices[0]]
        return results