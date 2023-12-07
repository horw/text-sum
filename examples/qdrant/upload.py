from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


f = PyPDFLoader('esp32_technical_reference_manual_en.pdf').load()
splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=10)
text = splitter.split_documents(f)


embedding = HuggingFaceEmbeddings() # in default use model_name = "sentence-transformers/all-mpnet-base-v2"
q = Qdrant.from_documents(
    documents=text,
    embedding=embedding,
    url='http://localhost:6333',
    collection_name='hello_world'
)
