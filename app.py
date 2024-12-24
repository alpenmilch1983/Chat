import os
from flask import Flask, request, jsonify, render_template

# Wichtig: PyPDFLoader in neueren Versionen i. d. R. in langchain_community
from langchain_community.document_loaders import PyPDFLoader

# Text-Splitter kommt aus langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma aus langchain_community
from langchain_community.vectorstores import Chroma

# RetrievalQA aus langchain_core.chains (in vielen aktuellen Versionen)
from langchain_core.chains import RetrievalQA

# OpenAIEmbeddings, ChatOpenAI aus langchain_openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


app = Flask(__name__, template_folder="templates", static_folder="static")

# 1. Lies deinen OpenAI-API-Key z. B. aus 'api.txt' im selben Ordner
with open("api.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 2. Verzeichnis mit den PDF-Dateien
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

# 3. PDF-Dokumente laden
documents = load_and_split_pdfs(PDF_DIRECTORY)

# 4. Embedding-Modell (Achtung: text-embedding-3-large ist ein Beispiel)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 5. Chroma-Vektorstore anlegen/aktualisieren
#    Falls du eine alte DB mit anderen Dimensionen hast, evtl. Ordner chroma_db löschen.
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()

# RetrievalQA-Kette
qa_chain = None

def init_qa_chain():
    global qa_chain
    # Hypothetisches Modell "gpt-4o-mini-2024-07-18" mit Temperatur 0.4
    llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o-mini-2024-07-18")
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
    """Empfängt eine Frage (`query`) im JSON-Body und gibt die Antwort zurück."""
    data = request.get_json()
    user_question = data.get("query", "")

    if qa_chain is None:
        init_qa_chain()

    result = qa_chain.run(user_question)
    return jsonify({"answer": result})

if __name__ == "__main__":
    # Kette vor dem Serverstart initialisieren
    init_qa_chain()
    app.run(host="0.0.0.0", port=5000, debug=True)
