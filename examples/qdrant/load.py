from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores.qdrant import Qdrant

embedding = HuggingFaceEmbeddings()

client = QdrantClient(
    url='http://localhost:6333',
)

db = Qdrant(
    client=client,
    embeddings=embedding,
    collection_name='hello_world'
)

res = db.similarity_search_with_score("how many GPIO esp32 have", k=5)
for t, s in res:
    print(s, t.page_content)
    print("PAGE  ", t.metadata['page'])



