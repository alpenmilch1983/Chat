import os
from flask import Flask, request, jsonify, render_template

# PyPDFLoader aus langchain_community.document_loaders
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter aus langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma aus langchain_community.vectorstores
from langchain_community.vectorstores import Chroma

# RetrievalQA aus langchain.chains
from langchain.chains import RetrievalQA

# OpenAIEmbeddings und ChatOpenAI aus langchain_openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")

# 1) OpenAI API-Key aus "api.txt" im selben Verzeichnis lesen
with open("api.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 2) Pfad zum PDF-Verzeichnis
PDF_DIRECTORY = "docs"

def load_and_split_pdfs(directory):
    """Liest PDFs aus einem Verzeichnis ein und zerlegt sie in Text-Chunks."""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  # Angepasste Chunk-Größe
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            try:
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    chunks = splitter.split_text(doc.page_content)
                    all_docs.extend(chunks)
            except Exception as e:
                print(f"Fehler beim Laden von {file_name}: {e}")
    return all_docs

# 3) Dokumente einlesen und splitten
documents = load_and_split_pdfs(PDF_DIRECTORY)

# 4) Embedding-Modell setzen (Hypothese: text-embedding-3-large existiert in deinem Setup)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 5) Chroma als Vektor-DB
#    Falls du vormals ein anderes Modell (andere Dimension) hattest, lösche ggf. "chroma_db"
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()

qa_chain = None

def init_qa_chain():
    """Initialisiert die RetrievalQA-Kette mit unserem Chat-Modell."""
    global qa_chain
    llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o-mini-2024-07-18")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # Simple "stuff" Chain
        retriever=retriever
    )

@app.route("/")
def index():
    """Startseite (z. B. mit deinem Chat-Frontend)"""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Empfängt die User-Frage und gibt die generierte Antwort als JSON zurück."""
    data = request.get_json()
    user_question = data.get("query", "")

    if qa_chain is None:
        init_qa_chain()

    # Verwende invoke und extrahiere die relevante Antwort
    result = qa_chain.invoke(user_question)

    # Prüfe die Struktur von result und extrahiere die Antwort
    if isinstance(result, dict):
        answer = result.get('output', 'Keine Antwort erhalten.')
    else:
        answer = str(result)  # Fallback, falls die Struktur anders ist

    return jsonify({"answer": answer})

if __name__ == "__main__":
    init_qa_chain()
    app.run(host="0.0.0.0", port=5000, debug=True)
