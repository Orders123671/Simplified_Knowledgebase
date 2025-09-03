import streamlit as st
import google.auth
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore as fa_firestore
import io
import pypdf
import uuid
from PIL import Image
import os
import pytesseract
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import torch
import numpy as np
import json

# --- UI Layout ---
st.set_page_config(page_title="Katrina Knowledgebase", layout="wide")

# Add custom CSS to style buttons
st.markdown("""
<style>
.stButton>button, .stFormSubmitButton>button {
    background-color: #ff1493;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Firestore Initialization ---
try:
    if not firebase_admin._apps:
        # Most reliable method: use a service account key from secrets
        if "firebase_credentials" in st.secrets:
            cred_info = json.loads(st.secrets["firebase_credentials"])
            cred = credentials.Certificate(cred_info)
            firebase_admin.initialize_app(cred, {'projectId': cred_info['project_id']})
        else:
            st.error("Firebase credentials not found in `.streamlit/secrets.toml`.")
            st.stop()
    db = fa_firestore.client()
except Exception as e:
    st.error(f"Error initializing Firestore: {e}")
    st.warning("Please ensure your Firestore credentials are correctly set up in `.streamlit/secrets.toml`.")
    st.stop()

# --- Gemini API and Embeddings Model Initialization ---
try:
    # The Gemini API key is now read from Streamlit's secrets file
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    
    # The SentenceTransformer model will be loaded and cached
    @st.cache_resource
    def get_embedding_model():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    embedding_model = get_embedding_model()
except KeyError:
    st.error("GEMINI_API_KEY not found in .streamlit/secrets.toml.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing AI components: {e}")
    st.stop()

# --- Helper Functions for File Processing and Embedding ---
def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_image(image_file):
    # Set the path to the Tesseract executable
    if 'TESSERACT_CMD' in os.environ:
        pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']
    
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error performing OCR: {e}. Please ensure Tesseract is installed and in your system's PATH.")
        return None

def extract_text_from_xlsx(xlsx_file):
    try:
        import openpyxl
        xls = pd.ExcelFile(xlsx_file)
        text = ""
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            text += df.to_string() + "\n\n"
        return text
    except ImportError:
        st.error("Error processing XLSX file: Missing optional dependency 'openpyxl'. Please install it by running: pip install openpyxl")
        return None
    except Exception as e:
        st.error(f"Error processing XLSX file: {e}")
        return None

def create_document_with_embedding(title, content, content_type):
    doc_id = str(uuid.uuid4())
    doc_ref = db.collection("knowledge_base").document(doc_id)

    # Generate the embedding vector for the content
    embedding = embedding_model.encode(content).tolist()
    
    doc_ref.set({
        "title": title,
        "content": content,
        "content_type": content_type,
        "embedding": embedding,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "uuid": doc_id
    })
    st.success(f"Successfully added '{title}' to the knowledge base!")

def get_conversational_response(user_query, retrieved_docs):
    """Generates a conversational response using Gemini, grounded in the provided context."""
    if not retrieved_docs:
        return "I am sorry, but I cannot find any relevant information in the knowledge base."

    context_list = []
    for doc in retrieved_docs:
        content = doc.get('content', '')
        content_type = doc.get('content_type', 'string')
        title = doc.get('title', 'Untitled Document')
        
        # Customize the context based on file type for better LLM grounding
        if content_type == 'xlsx':
            context_list.append(f"The following is from the spreadsheet titled '{title}'. Treat this as tabular data:\n{content}")
        else:
            context_list.append(f"The following is from the document titled '{title}':\n{content}")
    
    full_context = "\n\n".join(context_list)

    full_prompt = (
        f"You are a helpful knowledge base assistant. Answer the user's question based on the following context. "
        f"If the answer is not in the context, say 'I am sorry, but I cannot find the answer to your question in the knowledge base.'\n\n"
        f"**Context:**\n{full_context}\n\n"
        f"**User Question:**\n{user_query}"
    )
    
    # Use a try-except block to handle potential API errors
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "Sorry, I am unable to generate a response at this time."

def get_similar_documents(query_embedding):
    """Finds and retrieves documents with the most similar embeddings."""
    docs = db.collection("knowledge_base").stream()
    
    query_vector = np.array(query_embedding)
    similarities = []

    for doc in docs:
        doc_data = doc.to_dict()
        if 'embedding' in doc_data and doc_data['embedding'] is not None:
            doc_vector = np.array(doc_data['embedding'])
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((similarity, doc_data))
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Return top 3 most similar documents
    return [doc for score, doc in similarities[:3] if score > 0.5] # Filter by a minimum similarity score

def fallback_keyword_search(user_query):
    """Performs a simple keyword search as a fallback."""
    docs = db.collection("knowledge_base").stream()
    keywords = user_query.lower().split()
    results = []

    for doc in docs:
        doc_data = doc.to_dict()
        content = doc_data.get('content', '').lower()
        if any(keyword in content for keyword in keywords):
            results.append(doc_data)
    return results

def clear_search_history():
    st.session_state.messages = []

# --- Helper functions for deletion ---
@st.cache_data(show_spinner=False)
def get_all_documents():
    """Retrieves all documents from Firestore for deletion."""
    docs = db.collection("knowledge_base").stream()
    return [{"id": doc.id, "title": doc.to_dict().get("title", "Untitled")} for doc in docs]

def delete_document(doc_id):
    """Deletes a document from Firestore and clears the cache."""
    try:
        db.collection("knowledge_base").document(doc_id).delete()
        st.success(f"Successfully deleted document.")
        get_all_documents.clear() # Clear cache to force a refresh of the list
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting document: {e}")

# --- Sidebar for Admin Panel ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Admin Panel</h2>", unsafe_allow_html=True)
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.success("Access Granted.")
        
        st.markdown("<h4 style='text-align: center;'>Upload Files</h4>", unsafe_allow_html=True)
        st.markdown("Upload PDFs, JPEGs, or text to be saved in the knowledge base.")
        
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "jpeg", "jpg", "png", "txt", "csv", "xlsx"]
        )
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            content = None
            content_type = file_extension
            title = uploaded_file.name

            if file_extension == "pdf":
                content = extract_text_from_pdf(uploaded_file)
            elif file_extension in ["jpeg", "jpg", "png"]:
                content = extract_text_from_image(uploaded_file)
            elif file_extension in ["txt", "csv"]:
                content = uploaded_file.getvalue().decode("utf-8")
            elif file_extension == "xlsx":
                content = extract_text_from_xlsx(uploaded_file)
            
            if content:
                # Display the extracted content for debugging
                with st.expander("Show Extracted Content"):
                    st.text_area("Content from file:", value=content, height=300)
                
                create_document_with_embedding(title, content, content_type)

        st.markdown("<h4 style='text-align: center;'>Add Text</h4>", unsafe_allow_html=True)
        st.text_input("Title for text content:", key="text_title", label_visibility="hidden")
        st.text_area("Enter text content:", key="text_content", label_visibility="hidden")
    
        if st.button("**Save Text Content**"):
            if st.session_state.text_title and st.session_state.text_content:
                create_document_with_embedding(st.session_state.text_title, st.session_state.text_content, "string")
            else:
                st.warning("Please provide both a title and content for the text.")
        
        st.markdown("---")
        st.markdown("<h4 style='text-align: center;'>Delete Data</h4>", unsafe_allow_html=True)
        all_docs = get_all_documents()
        if all_docs:
            doc_options = {f"{doc['title']} (ID: {doc['id']})": doc['id'] for doc in all_docs}
            selected_option = st.selectbox("Select a document to delete:", list(doc_options.keys()))
            selected_doc_id = doc_options[selected_option]

            if st.button("**Delete Selected Document**"):
                delete_document(selected_doc_id)
        else:
            st.info("No documents found to delete.")

        st.button("**Logout**", on_click=lambda: st.session_state.update(logged_in=False, messages=[]))

    else:
        st.markdown("<h5 style='text-align: center;'>Enter admin password:</h5>", unsafe_allow_html=True)
        password = st.text_input("", type="password", label_visibility="hidden")
        if st.button("**Login**"):
            if password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect password.")

# --- Main App Logic ---
st.markdown("<h1 style='text-align: center; color: #ff1493;'><b>Katrina Knowledgebase 🎂</b></h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Ask the Knowledge Base</h2>", unsafe_allow_html=True)
st.markdown("---")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Place the chat input in a centered column
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if user_query := st.chat_input("Ask a question about the knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
    
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                query_embedding = embedding_model.encode(user_query).tolist()
                retrieved_docs = get_similar_documents(query_embedding)
                
                # Fallback to keyword search if no relevant documents are found
                if not retrieved_docs:
                    retrieved_docs = fallback_keyword_search(user_query)

                response_text = get_conversational_response(user_query, retrieved_docs)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.button("**Clear Chat**", on_click=clear_search_history)
