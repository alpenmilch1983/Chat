import os
from flask import Flask, request, jsonify, render_template

# Statt langchain_community.document_loaders oder langchain.document_loaders
# kann PyPDFLoader jetzt in "langchain_core.document_loaders" liegen. (Siehe Migrationshinweise)
from langchain_core.document_loaders import PyPDFLoader

# Statt langchain.text_splitter -> langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Statt langchain.vectorstores -> langchain_community.vectorstores
from langchain_community.vectorstores import Chroma

# Statt langchain.chains -> langchain_core.chains
from langchain_core.chains import RetrievalQA

# Statt langchain.embeddings.openai / langchain.chat_models -> langchain_openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Flask initialisieren
app = Flask(__name__, template_folder="templates", static_folder="static")

# Lies deinen OpenAI-API-Key z. B. aus 'api.txt' im selben Ordner
with open("api.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Pfad zu deinem PDF-Verzeichnis
PDF_DIRECTORY = "docs"

def load_and_split_pdfs(directory):
    """Liest alle PDFs aus dem Ordner ein und zerlegt sie in Text-Chunks."""
    all_docs = []
    # Neu: RecursiveCharacterTextSplitter kommt jetzt aus langchain_text_splitters
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            # PyPDFLoader jetzt aus langchain_core.document_loaders
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                chunks = splitter.split_text(doc.page_content)
                for chunk in chunks:
                    all_docs.append(chunk)
    return all_docs

# Dokumente einlesen
documents = load_and_split_pdfs(PDF_DIRECTORY)

# OpenAIEmbeddings nun aus langchain_openai
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Chroma kommt aus langchain_community.vectorstores
# Achte darauf, dass du ggf. dein persist_directory löscht/änderst, 
# falls die Dimension nicht zusammenpasst (DimensionMismatch).
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()
qa_chain = None

def init_qa_chain():
    global qa_chain
    # RetrievalQA kommt nun aus langchain_core.chains
    llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o-mini-2024-07-18")
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
