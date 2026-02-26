class Retriever:
    def __init__(self, embedder, index_builder):
        self.embedder = embedder
        self.index_builder = index_builder

    def retrieve(self, claim, top_k=3):
        query_embedding = self.embedder.embed([claim])
        return self.index_builder.search(query_embedding, top_k)