import os
from flask import Flask, request, jsonify, render_template

# PDF-Loader – je nach LangChain-Version: langchain_community oder direkt langchain
from langchain_community.document_loaders import PyPDFLoader  
# from langchain.document_loaders import PyPDFLoader  # (Bei älteren LC-Versionen)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Neuere Imports aus langchain_openai statt langchain.embeddings/chat_models
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")

# 1. API-Key aus Datei 'api.txt' lesen
with open("api.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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

# 2. Dokumente einlesen
documents = load_and_split_pdfs(PDF_DIRECTORY)

# 3. Embeddings mit 'text-embedding-3-large' erzeugen
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 4. Chroma-Datenbank neu anlegen / aktualisieren
#    Falls eine alte DB existiert (z. B. mit anderer Dimension), lösche vorher das Verzeichnis 'chroma_db'
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="chroma_db"  
)

retriever = vectorstore.as_retriever()

# 5. Chain-Initialisierung (z. B. RetrievalQA)
qa_chain = None
def init_qa_chain():
    global qa_chain
    from langchain.chains import RetrievalQA
    # Hier verwenden wir das hypothetische Chat-Modell 'gpt-4o-mini-2024-07-18'
    # Bitte sicherstellen, dass dieser Modellname bei deinem Anbieter existiert/erlaubt ist.
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("query", "")

    if qa_chain is None:
        init_qa_chain()

    result = qa_chain.run(user_question)
    return jsonify({"answer": result})

if __name__ == "__main__":
    init_qa_chain()
    app.run(host="0.0.0.0", port=5000, debug=True)
