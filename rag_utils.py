import PyPDF2
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file):
    """Extract text using pdfplumber (more robust)"""
    try:
        pdf_file.seek(0)
        text = ""

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) < 50:
            raise Exception("Could not extract meaningful text from PDF")

        return text

    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks


def create_embedding(text):
    """Create vector embedding for text - returns list for pgvector"""
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def find_relevant_chunks(query, all_chunks_with_embeddings, top_k=3):
    """Find most relevant chunks for a query using cosine similarity"""
    query_embedding = embedding_model.encode(query)

    similarities = []
    for chunk_data in all_chunks_with_embeddings:
        try:
            chunk_embedding = np.array(chunk_data["embedding"])

            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )

            similarities.append(
                {
                    "chunk_text": chunk_data["chunk_text"],
                    "similarity": float(similarity),
                    "document_id": chunk_data.get("document_id"),
                    "chunk_index": chunk_data.get("chunk_index"),
                }
            )
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]

