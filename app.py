import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader  # Achtung: ggf. from langchain.document_loaders import PyPDFLoader, wenn du eine ältere Version verwendest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Flask-App initialisieren
app = Flask(__name__)

# 2. OpenAI API Key (ersetzen durch deinen eigenen Schlüssel)
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 3. Pfad zum Ordner mit den PDF-Dateien
PDF_DIRECTORY = "docs"

# 4. PDF-Dateien einlesen und in Text-Dokumente splitten
def load_and_split_pdfs(directory):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                # In kleinere Chunks aufteilen
                chunks = splitter.split_text(doc.page_content)
                for chunk in chunks:
                    all_docs.append(chunk)
    return all_docs

# 5. Einlesen der Dokumente
documents = load_and_split_pdfs(PDF_DIRECTORY)

# 6. Embeddings erzeugen und in Chroma-Datenbank ablegen
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Beispielhaftes Embedding-Modell
vectorstore = Chroma.from_texts(documents, embedding=embeddings, persist_directory="chroma_db")

# 7. RetrievalQA-Chain aufsetzen, jetzt mit "gpt-3.5-turbo"
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

@app.route("/ask", methods=["POST"])
def ask():
    # Anfrage (Query) aus dem Request-Body
    data = request.get_json()
    user_question = data.get("query", "")

    # Query an das RetrievalQA-System senden
    result = qa_chain.run(user_question)
    
    # Ergebnis als JSON zurückgeben
    return jsonify({"answer": result})

if __name__ == "__main__":
    # Flask-Server starten
    app.run(host="0.0.0.0", port=5000, debug=True)
