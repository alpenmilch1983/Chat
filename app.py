import os
from flask import Flask, request, jsonify, render_template

# Falls du eine neuere LangChain-Version hast und PyPDFLoader in langchain_community liegt:
from langchain_community.document_loaders import PyPDFLoader  
# (Bei älteren Versionen evtl.: from langchain.document_loaders import PyPDFLoader)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Neu: Klasse OpenAIEmbeddings & ChatOpenAI kommen aus langchain_openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Flask-Initialisierung
app = Flask(__name__, template_folder="templates", static_folder="static")

# 1. Lies deinen OpenAI API Key aus der Datei "api.txt"
with open("api.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()

# 2. Setze den API Key in der Umgebung (kann von den OpenAI-Klassen genutzt werden)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Pfad zu deinem Dokumenten-Verzeichnis (mit PDF-Dateien)
PDF_DIRECTORY = "docs"

def load_and_split_pdfs(directory):
    """Liest alle PDFs aus dem Ordner ein und zerlegt sie in Text-Chunks."""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                chunks = splitter.split_text(doc.page_content)
                for chunk in chunks:
                    all_docs.append(chunk)
    return all_docs

# Dokumente verarbeiten
documents = load_and_split_pdfs(PDF_DIRECTORY)

# Embeddings mit dem neuen Paket (Achte auf das Modell; "text-embedding-ada-002" ist Standard)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Chroma-Datenbank erzeugen (oder laden, falls persist_directory bereits existiert)
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Retrieval-Objekt
retriever = vectorstore.as_retriever()
qa_chain = None  # Wird beim ersten Aufruf initialisiert

def init_qa_chain():
    """Erzeugt die RetrievalQA-Kette mit ChatOpenAI."""
    global qa_chain
    from langchain.chains import RetrievalQA
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

@app.route("/")
def index():
    """Zeigt die Startseite mit dem Chatfenster (index.html)."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Empfängt eine Frage (`query`) im JSON-Body und gibt die Antwort als JSON zurück."""
    data = request.get_json()
    user_question = data.get("query", "")

    if qa_chain is None:
        init_qa_chain()

    result = qa_chain.run(user_question)
    return jsonify({"answer": result})

if __name__ == "__main__":
    # QA-Chain schon beim Start initialisieren
    init_qa_chain()
    # Server starten
    app.run(host="0.0.0.0", port=5000, debug=False)
