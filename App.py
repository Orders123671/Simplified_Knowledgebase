import streamlit as st
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
import numpy as np
import json
import docx
import base64
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from streamlit_option_menu import option_menu

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
db = None  # Initialize db to None to prevent NameError
try:
    if not firebase_admin._apps:
        # Check for local credentials first, then for cloud secrets
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        elif "firebase" in st.secrets:
            cred = credentials.Certificate(st.secrets["firebase"])
            firebase_admin.initialize_app(cred)
        else:
            st.error(f"Error initializing Firestore: No credentials found.")
            st.warning("Please ensure your Firestore credentials are correctly set up in `serviceAccountKey.json` for local run or `.streamlit/secrets.toml` for cloud deployment.")
            st.stop()
    # Ensure the db client is always assigned, regardless of whether the app is being initialized.
    db = fa_firestore.client()
except Exception as e:
    st.error(f"Error initializing Firestore: {e}")
    st.warning("Please ensure your Firestore credentials are correctly set up.")
    st.stop()

# --- Gemini API and Embeddings Model Initialization ---
try:
    # The Gemini API key is now read from Streamlit's secrets file
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    
    # Define a new multi-modal embedding function
    @st.cache_resource
    def get_embedding(content):
        # This function now only handles text for embedding, which is the correct use-case
        return genai.embed_content(model="models/embedding-001",
                                   content=content,
                               _content_type="retrieval_query")["embedding"]
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

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing DOCX file: {e}")
        return None

def create_document_with_embedding(title, content, content_type, image_data=None):
    doc_id = str(uuid.uuid4())
    doc_ref = db.collection("knowledge_base").document(doc_id)
    
    # Generate the embedding vector for the content
    embedding = get_embedding(content)
    
    data_to_store = {
        "title": title,
        "content": content,
        "content_type": content_type,
        "embedding": embedding,
        "timestamp": fa_firestore.SERVER_TIMESTAMP,
        "uuid": doc_id
    }

    if image_data:
        # Resize image before saving to stay within Firestore document size limits
        # Decode the Base64 string back into bytes before opening with PIL
        image_bytes_decoded = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes_decoded))
        max_size = (500, 500)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert resized image back to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        data_to_store["image_data"] = image_data
    
    doc_ref.set(data_to_store)
    st.success(f"Successfully added '{title}' to the knowledge base!")

def get_conversational_response(user_query, retrieved_docs):
    if not retrieved_docs:
        return "I am sorry, but I cannot find any relevant information in the knowledge base."

    context_list = []
    for doc in retrieved_docs:
        content = doc.get('content', '')
        content_type = doc.get('content_type', 'string')
        title = doc.get('title', 'Untitled Document')
        
        if content_type == 'xlsx':
            context_list.append(f"The following is from the spreadsheet titled '{title}'. Treat this as tabular data:\n{content}")
        else:
            context_list.append(f"The following is from the document titled '{title}':\n{content}")
    
    full_context = "\n\n".join(context_list)

    full_prompt = (
        f"You are a helpful knowledge base assistant. You have retrieved the following information which may or may not be relevant to the user's query. "
        f"Please provide a concise and helpful response based on the retrieved information, even if it is only a partial match. If the information is not relevant at all, say so."
        f"**Context:**\n{full_context}\n\n"
        f"**User Question:**\n{user_query}"
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        return response.text
    except ResourceExhausted:
        return "Sorry, I have exceeded my daily quota and cannot generate a response at this time. Please try again later."
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "Sorry, I am unable to generate a response at this time."

def get_similar_documents(query_embedding):
    docs = db.collection("knowledge_base").stream()
    
    query_vector = np.array(query_embedding)
    similarities = []

    for doc in docs:
        doc_data = doc.to_dict()
        if 'embedding' in doc_data and doc_data['embedding'] is not None:
            doc_vector = np.array(doc_data['embedding'])
            
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((similarity, doc_data))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in similarities[:3] if score > 0.5]

def fallback_keyword_search(user_query):
    docs = db.collection("knowledge_base").stream()
    keywords = user_query.lower().split()
    results = []

    for doc in docs:
        doc_data = doc.to_dict()
        content = doc_data.get('content', '').lower()
        if any(keyword in content for keyword in keywords):
            results.append(doc_data)
    return results

def clear_chat_history():
    st.session_state.messages = []

@st.cache_data(show_spinner=False)
def get_all_documents():
    docs = db.collection("knowledge_base").stream()
    return [{"id": doc.id, "title": doc.to_dict().get("title", "Untitled")} for doc in docs]

def delete_multiple_documents(doc_ids):
    """Deletes multiple documents from Firestore and clears the cache."""
    try:
        for doc_id in doc_ids:
            db.collection("knowledge_base").document(doc_id).delete()
        st.success(f"Successfully deleted {len(doc_ids)} documents.")
        get_all_documents.clear() # Clear cache to force a refresh of the list
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting documents: {e}")
        
def get_image_description(image):
    """Uses a vision model to generate a text description of an image."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(["Describe this image to use as a search query.", image])
        return response.text
    except Exception as e:
        st.error(f"Error generating image description with Gemini: {e}")
        return "Could not describe the image."

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
            type=["pdf", "jpeg", "jpg", "png", "txt", "csv", "xlsx", "docx"]
        )
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            content = None
            content_type = file_extension
            title = uploaded_file.name
            image_data = None

            if file_extension == "pdf":
                content = extract_text_from_pdf(uploaded_file)
            elif file_extension in ["jpeg", "jpg", "png"]:
                image_bytes = uploaded_file.getvalue()
                content = extract_text_from_image(io.BytesIO(image_bytes))
                if content:
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
            elif file_extension in ["txt", "csv"]:
                content = uploaded_file.getvalue().decode("utf-8")
            elif file_extension == "xlsx":
                content = extract_text_from_xlsx(uploaded_file)
            elif file_extension == "docx":
                content = extract_text_from_docx(uploaded_file)
            
            if content:
                with st.expander("Show Extracted Content"):
                    st.text_area("Content from file:", value=content, height=300)
                
                create_document_with_embedding(title, content, content_type, image_data)

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
            selected_options = st.multiselect("Select documents to delete:", list(doc_options.keys()))
            selected_doc_ids = [doc_options[option] for option in selected_options]

            if st.button("**Delete Selected Documents**"):
                if selected_doc_ids:
                    delete_multiple_documents(selected_doc_ids)
                else:
                    st.warning("Please select at least one document to delete.")
        else:
            st.info("No documents found to delete.")

        st.button("**Logout**", on_click=lambda: st.session_state.update(logged_in=False, messages=[]))

    else:
        st.markdown("<h5 style='text-align: center;'>Admin Login</h5>", unsafe_allow_html=True)
        password = st.text_input("", type="password", label_visibility="hidden")
        if st.button("**Login**"):
            if password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect password.")

# --- Main App Logic ---
st.markdown("<h1 style='text-align: center; color: #ff1493;'><b>Katrina Knowledgebase 🎂</b></h1>", unsafe_allow_html=True)
st.markdown("---")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if 'image' in message:
            st.image(message['image'], caption="User Upload")
        st.markdown(message["content"])

# Multi-modal input
st.markdown("**Search the Knowledge Base**")

# This form is used to prevent the `Clear Chat` button from triggering a search
with st.form(key='chat_form'):
    uploaded_image = st.file_uploader("Or, search using an image...", type=["jpeg", "jpg", "png"], key="image_uploader")
    user_query = st.text_input("Ask a question about the knowledge base...", key="user_query_input")
    submit_button = st.form_submit_button("Search")

if submit_button and (user_query or uploaded_image):
    with st.spinner("Searching and generating response..."):
        if uploaded_image:
            uploaded_image_pil = Image.open(uploaded_image)
            image_description = get_image_description(uploaded_image_pil)
            
            st.session_state.messages.append({"role": "user", "content": "uploaded an image.", "image": uploaded_image_pil})
            with st.chat_message("user"):
                st.image(uploaded_image_pil, caption="User Upload")
                st.markdown(f"**Image description:** {image_description}")
                
            query_embedding = get_embedding(image_description)
            search_query = image_description
        
        elif user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            query_embedding = get_embedding(user_query)
            search_query = user_query
        
        with st.chat_message("assistant"):
            retrieved_docs = get_similar_documents(query_embedding)
            
            if not retrieved_docs and 'search_query' in locals():
                retrieved_docs = fallback_keyword_search(search_query)

            for doc in retrieved_docs:
                if 'image_data' in doc and doc['image_data']:
                    try:
                        image_bytes = base64.b64decode(doc['image_data'])
                        st.image(image_bytes, caption=doc.get('title', 'Uploaded Image'))
                    except Exception as e:
                        st.error(f"Could not display image: {e}")
            
            response_text = get_conversational_response(search_query, retrieved_docs)
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

st.button("Clear Chat", on_click=lambda: st.session_state.update(messages=[]), key="clear_chat_button")
