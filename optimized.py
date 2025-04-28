import os
import streamlit as st
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.embeddings import Embeddings

# 1. Dynamically quantize your embedding model (cached)
@st.cache_resource(ttl=3600)
def load_quantized_model():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

class QuantizedEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# 2. Lightweight “ping” endpoint to mitigate sleeping
if "ping" in st.query_params:
    st.write("pong")
    st.stop()

# 3. Load API key
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY! Add it to .env or Streamlit Secrets.")
    st.stop()

st.title("Conversational RAG With PDF Uploads and Chat History")

# 4. Cache PDF loading/splitting
@st.cache_data(max_entries=3, persist="disk")
def load_and_process_pdfs(pdf_paths: list[str]):
    docs = []
    for path in pdf_paths:
        docs.extend(PyPDFLoader(path).load())
    return docs

# 5. Cache FAISS index creation
@st.cache_resource(ttl=3600)
def generate_vectorstore(_docs: list):
    splits = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500
    ).split_documents(_docs)
    return FAISS.from_documents(splits, QuantizedEmbeddings(load_quantized_model()))

# 6. Cache your LLM client (remove stop sequence so it actually speaks)
@st.cache_resource(ttl=3600)
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="deepseek-r1-distill-llama-70b",
    )

# 7. Initialize resources
documents = load_and_process_pdfs(["Health Montoring Box (CHATBOT).pdf"])
vectorstore = generate_vectorstore(documents)
retriever = vectorstore.as_retriever()
llm = load_llm()

# 8. Build RAG chain with revised system prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history + latest user question, reformulate into a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are a medical assistant. Answer in 1–2 sentences, "
    "include brief advice or abnormal ranges when relevant, "
    "and do not output any chain-of-thought or `<think>` tags.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in st.session_state:
        st.session_state[sid] = ChatMessageHistory()
    return st.session_state[sid]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# 9. Use your original extractor to strip out THINK blocks
def extract_final_answer(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()

# === UI ===
session_id = st.text_input("Session ID", "default_session")
user_input = st.text_input("Your question:", key="user_input")

if st.button("Submit"):
    if user_input:
        with st.spinner("Generating response..."):
            hist = get_session_history(session_id)
            out = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            raw = out["answer"]
            answer = extract_final_answer(raw)
            st.markdown(f"**Answer:** {answer}")
            with st.expander("Chat History"):
                st.write(hist.messages)

if st.button("Clear Chat History"):
    st.session_state[session_id] = ChatMessageHistory()
    st.success("Chat history cleared!")
