import os
from flask import Flask, request, jsonify, render_template
# Falls du 'langchain_community' brauchst:
# from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")

OPENAI_API_KEY = "DEIN_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PDF_DIRECTORY = "docs"

def load_and_split_pdfs(directory):
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

documents = load_and_split_pdfs(PDF_DIRECTORY)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_texts(documents, embedding=embeddings, persist_directory="chroma_db")

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

@app.route("/")
def index():
    # Die Startseite mit dem Chatfenster
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("query", "")
    result = qa_chain.run(user_question)
    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
