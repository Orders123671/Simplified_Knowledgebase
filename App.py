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
import bcrypt
import mimetypes
from google.cloud import aiplatform, storage
# BUG FIX: Use the direct, versioned import for IndexDatapoint
from google.cloud.aiplatform_v1.types import IndexDatapoint
# BUG FIX: Import the service_account module to handle credentials explicitly
from google.oauth2 import service_account
import ast # Safely parse string representations of Python literals
import time # Import the time module for the sequential display effect

# --- UI Layout ---
LOGO_PATH = "Katrina_logo.png"
page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🎂"
st.set_page_config(page_title="Katrina Knowledgebase", page_icon=page_icon, layout="wide")

# Add custom CSS to style buttons and chat container
st.markdown("""
<style>
    .stButton>button, .stFormSubmitButton>button {
        background-color: #ff1493;
        color: white;
        font-weight: bold;
    }
    .chat-container {
        border: 2px solid #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        color: #31333F;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- Optimized Service Initialization ---
@st.cache_resource
def init_services():
    """
    Initializes and caches connections to all backend services. This function uses a robust
    authentication method that works for both local development and Streamlit Cloud deployment.
    """
    creds_dict = None
    # --- Unified Credential Handling ---
    try:
        if os.path.exists("serviceAccountKey.json"):
            with open("serviceAccountKey.json") as f:
                creds_dict = json.load(f)
        elif "firebase" in st.secrets:
            firebase_creds = st.secrets["firebase"]
            if isinstance(firebase_creds, str):
                creds_dict = ast.literal_eval(firebase_creds)
            else:
                creds_dict = dict(firebase_creds)
        else:
            st.error("Error: No Google Cloud credentials found in local file or secrets.")
            st.stop()

        gcp_credentials = service_account.Credentials.from_service_account_info(creds_dict)

        # --- Firestore Initialization ---
        if not firebase_admin._apps:
            cred_obj_for_firebase = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred_obj_for_firebase)
        db = fa_firestore.client()

        # --- Google Cloud Storage Initialization ---
        storage_client = storage.Client(credentials=gcp_credentials)
        gcs_bucket_name = st.secrets["GCP_STORAGE_BUCKET"]
        bucket = storage_client.bucket(gcs_bucket_name)

        # --- Vertex AI Vector Search Initialization ---
        gcp_project_id = st.secrets["GCP_PROJECT_ID"]
        gcp_region = st.secrets["GCP_REGION"]
        vertex_ai_index_id = st.secrets["VERTEX_AI_INDEX_ID"]
        vertex_ai_endpoint_id = st.secrets["VERTEX_AI_ENDPOINT_ID"]
        if "VERTEX_AI_DEPLOYED_INDEX_ID" not in st.secrets:
            st.error("Configuration missing: Please add VERTEX_AI_DEPLOYED_INDEX_ID.")
            st.stop()

        full_index_name = f"projects/{gcp_project_id}/locations/{gcp_region}/indexes/{vertex_ai_index_id}"
        
        aiplatform.init(project=gcp_project_id, location=gcp_region, credentials=gcp_credentials)
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=vertex_ai_endpoint_id)

        # --- Gemini API and Embeddings Model Initialization ---
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=gemini_api_key)
        
    except Exception as e:
        st.error(f"Error during service initialization: {e}")
        st.stop()
        
    return db, index_endpoint, full_index_name, bucket

# Initialize all services at once and cache the result
db, index_endpoint, full_index_name, gcs_bucket = init_services()


# --- Gemini API and Embeddings Model Initialization ---
@st.cache_resource
def get_embedding(content):
    return genai.embed_content(model="models/embedding-001",
                                 content=content,
                                 task_type="retrieval_query")["embedding"]

# --- User Authentication Functions ---
@st.cache_data(ttl=300)
def get_user_count():
    users_ref = db.collection("users")
    return len(list(users_ref.stream()))

@st.cache_data(show_spinner=False)
def get_all_users():
    users_ref = db.collection("users").stream()
    return [{"id": doc.id, "email": doc.to_dict().get("email", "N/A")} for doc in users_ref]

def delete_users(user_ids):
    try:
        for user_id in user_ids:
            db.collection("users").document(user_id).delete()
        st.success(f"Successfully deleted {len(user_ids)} user(s).")
        get_all_users.clear()
        get_user_count.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting users: {e}")

def register_user(email, password):
    if not email or not password:
        return "Email and password are required.", False
    
    if get_user_count() >= 50:
        return "User limit (50) reached. Cannot register new users.", False

    users_ref = db.collection("users")
    if users_ref.where("email", "==", email).get():
        return "An account with this email already exists.", False
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_ref.add({"email": email, "password_hash": hashed_password, "is_admin": False})
    get_user_count.clear()
    get_all_users.clear()
    return f"User '{email}' registered successfully!", True

def check_user(email, password):
    if not email or not password:
        return None, False
    users_ref = db.collection("users")
    query = users_ref.where("email", "==", email).limit(1).get()
    
    if not query:
        return None, False
        
    user_doc = query[0]
    user_data = user_doc.to_dict()
    
    if bcrypt.checkpw(password.encode('utf-8'), user_data.get("password_hash", b'')):
        is_admin = user_data.get("is_admin", False)
        return user_doc.id, is_admin
    return None, False

def check_admin(email, password):
    try:
        if email == st.secrets["ADMIN_EMAIL"] and password == st.secrets["ADMIN_PASSWORD"]:
            return True
    except KeyError:
        return False
    return False

# --- Helper Functions for File Processing and Embedding ---
def get_mime_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def text_chunker(text, chunk_size=1000, chunk_overlap=200):
    """Splits a long text into smaller chunks."""
    if text is None or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

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

def get_structured_chunks_from_xlsx(xlsx_file):
    """
    Extracts data from an Excel file and treats each row as a distinct chunk.
    """
    try:
        import openpyxl
        xls = pd.ExcelFile(xlsx_file)
        chunks = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            for index, row in df.iterrows():
                row_sentence = ", ".join([f"{col} is {val}" for col, val in row.items() if pd.notna(val)])
                if row_sentence:
                    chunks.append(f"In sheet '{sheet_name}', record {index+1}: {row_sentence}.")
        return chunks
    except ImportError:
        st.error("Error processing XLSX file: Missing optional dependency 'openpyxl'.")
        return []
    except Exception as e:
        st.error(f"Error processing XLSX file: {e}")
        return []

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

def create_document_with_embedding(title, filename, chunks, content_type, image_data=None, original_file_bytes=None):
    doc_id = str(uuid.uuid4())
    main_doc_ref = db.collection("knowledge_base").document(doc_id)
    
    download_url = None
    if original_file_bytes:
        try:
            blob = gcs_bucket.blob(f"{doc_id}_{filename}")
            blob.upload_from_string(original_file_bytes, content_type=get_mime_type(filename))
            download_url = blob.public_url
        except Exception as e:
            st.error(f"Failed to upload file to Google Cloud Storage: {e}")
            return

    data_to_store = {
        "title": title, "filename": filename, "content_type": content_type,
        "timestamp": fa_firestore.SERVER_TIMESTAMP, "uuid": doc_id, "download_url": download_url
    }

    if image_data:
        image_bytes_decoded = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes_decoded))
        img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        data_to_store["image_data"] = base64.b64encode(buffered.getvalue()).decode('utf-8')

    main_doc_ref.set(data_to_store)

    if not chunks:
        st.warning("Could not extract any text to index from the document.")
        return

    batch = db.batch()
    datapoints_for_vertex = []
    chunks_collection_ref = main_doc_ref.collection("chunks")

    for i, chunk_text in enumerate(chunks):
        chunk_doc_id = f"{doc_id}::{i}"
        embedding = get_embedding(chunk_text)
        
        chunk_doc_ref = chunks_collection_ref.document(str(i))
        batch.set(chunk_doc_ref, {"text": chunk_text})

        datapoints_for_vertex.append(IndexDatapoint(
            datapoint_id=chunk_doc_id,
            feature_vector=embedding
        ))
    
    try:
        batch.commit()
        index = aiplatform.MatchingEngineIndex(index_name=full_index_name)
        index.upsert_datapoints(datapoints=datapoints_for_vertex)
        st.success(f"Successfully chunked, indexed, and added '{title}' to the knowledge base!")
        get_all_documents.clear()
    except Exception as e:
        st.error(f"Failed during batch processing: {e}")
        main_doc_ref.delete()
        st.warning(f"Removed '{title}' from Firestore due to indexing failure.")

def get_conversational_response_stream(user_query, retrieved_docs, external_info=None):
    if not retrieved_docs and not external_info:
        yield "I am sorry, but I cannot find any relevant information."
        return

    context_prepend = f"**Real-time Information from the Web:**\n{external_info}\n\n---\n\n" if external_info else ""
    
    context_list = []
    for doc in retrieved_docs:
        title = doc.get('title', 'Untitled Document')
        content = doc.get('chunk_text', '')
        prefix = f"From the document titled '{title}':\n"
        context_list.append(f"{prefix}{content}")

    full_context = "\n\n".join(context_list)
    prompt = (
        "You are a helpful knowledge base assistant. Please provide a concise and helpful response based on ALL the information provided. "
        "Synthesize the information to give a complete answer. If no information is relevant at all, say so.\n\n"
        f"{context_prepend}"
        f"**Information from the Knowledge Base:**\n{full_context}\n\n"
        f"**User's Original Question:**\n{user_query}"
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response_stream = model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        yield "Sorry, I am unable to generate a response at this time."


def get_similar_documents(query_embedding):
    try:
        deployed_index_id = st.secrets["VERTEX_AI_DEPLOYED_INDEX_ID"]
        response = index_endpoint.find_neighbors(
            queries=[query_embedding],
            deployed_index_id=deployed_index_id,
            num_neighbors=3
        )
        if not response or not response[0]:
            return []

        final_docs = []
        for neighbor in response[0]:
            try:
                doc_id, chunk_index = neighbor.id.split("::")
                main_doc_ref = db.collection("knowledge_base").document(doc_id)
                chunk_doc_ref = main_doc_ref.collection("chunks").document(chunk_index)
                
                main_doc = main_doc_ref.get()
                chunk_doc = chunk_doc_ref.get()

                if main_doc.exists and chunk_doc.exists:
                    doc_data = main_doc.to_dict()
                    doc_data['chunk_text'] = chunk_doc.to_dict().get('text', '')
                    final_docs.append(doc_data)
            except (ValueError, IndexError):
                print(f"Could not parse neighbor ID: {neighbor.id}")
                continue
        
        return final_docs
    except Exception as e:
        st.error(f"Error querying Vertex AI Vector Search: {e}")
        return []

def fallback_keyword_search(user_query):
    return []

@st.cache_data(ttl=600)
def get_external_information(user_query):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro', tools=[{"google_search": {}}])
        prompt = (f"Based on a real-time Google Search, find the address, a Google Maps link, and a concise description for the query. Query: '{user_query}'")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text and "couldn't find" not in response.text.lower() else None
    except Exception as e:
        print(f"Error fetching external information: {e}")
        return None

@st.cache_data(ttl=600)
def get_intelligent_search_query(user_query):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (f"Rephrase the user's query to be more effective for a semantic search. Focus on the user's likely intent. Original query: '{user_query}'\nRephrased query:")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text and response.text.strip() else user_query
    except Exception as e:
        print(f"Error in get_intelligent_search_query: {e}")
        return user_query

@st.cache_data(ttl=600)
def get_spelling_suggestion(user_query):
    """
    Uses a generative model to correct spelling and suggest a clearer query.
    """
    if not user_query or len(user_query.split()) < 2:
        return user_query
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "You are a helpful spelling and grammar assistant. Correct any spelling mistakes in the following user query "
            "and return only the corrected phrase without any preamble. If the query is already correct, return it unchanged. "
            f"Original query: '{user_query}'\nCorrected query:"
        )
        response = model.generate_content(prompt)
        corrected_query = response.text.strip().strip('"')
        if len(corrected_query.split()) > len(user_query.split()) * 1.5:
             return user_query
        return corrected_query
    except Exception as e:
        print(f"Error in get_spelling_suggestion: {e}")
        return user_query

def clear_search():
    st.session_state.search_query = None
    if 'search_image' in st.session_state:
        del st.session_state.search_image

@st.cache_data(show_spinner="Refreshing document list...")
def get_all_documents():
    docs = db.collection("knowledge_base").stream()
    return [{"id": doc.id, "title": doc.to_dict().get("title", "Untitled"), "filename": doc.to_dict().get("filename", "Unknown")} for doc in docs]

def delete_multiple_documents(doc_ids_and_filenames):
    try:
        index = aiplatform.MatchingEngineIndex(index_name=full_index_name)
        datapoint_ids_to_delete = []

        for doc_id, filename in doc_ids_and_filenames:
            blob = gcs_bucket.blob(f"{doc_id}_{filename}")
            if blob.exists():
                blob.delete()

            chunks_ref = db.collection("knowledge_base").document(doc_id).collection("chunks")
            for chunk_doc in chunks_ref.stream():
                datapoint_ids_to_delete.append(f"{doc_id}::{chunk_doc.id}")
                chunk_doc.reference.delete()

            db.collection("knowledge_base").document(doc_id).delete()

        if datapoint_ids_to_delete:
            index.remove_datapoints(datapoint_ids=datapoint_ids_to_delete)

        st.success(f"Successfully deleted {len(doc_ids_and_filenames)} document(s) and all associated data.")
        get_all_documents.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Error during deletion: {e}")
        
def get_image_description(image):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(["Describe this image to use as a search query.", image])
        return response.text
    except Exception as e:
        st.error(f"Error generating image description with Gemini: {e}")
        return "Could not describe the image."

def get_image_as_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- Initialize session state ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'email' not in st.session_state:
    st.session_state.email = None


# --- Sidebar UI ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.image("https://placehold.co/250x100/FF1493/FFFFFF?text=Katrina&font=raleway", use_container_width=True)

    if st.session_state.user_id and not st.session_state.is_admin:
        st.markdown("<h2 style='text-align: center;'>User Portal</h2>", unsafe_allow_html=True)
        st.success(f"Logged in as {st.session_state.email}")
        st.info("You can now use the search assistant.")
        if st.button("**Logout**", key="user_logout_button"):
            logout_message = f"Logging out {st.session_state.email}..."
            with st.spinner(logout_message):
                st.session_state.clear()
            st.rerun()

    elif st.session_state.is_admin:
        st.markdown("<h2 style='text-align: center;'>Admin Portal</h2>", unsafe_allow_html=True)
        st.success(f"Logged in as Admin ({st.session_state.email})")
        
        tab1, tab2 = st.tabs(["User Management", "Knowledge Base Management"])

        with tab1:
            st.markdown("<h4 style='text-align: center;'>Register New User</h4>", unsafe_allow_html=True)
            st.info(f"Current Users: {get_user_count()} / 50")
            with st.form("register_form", clear_on_submit=True):
                new_email = st.text_input("New User Email")
                new_password = st.text_input("New User Password", type="password")
                if st.form_submit_button("Register User"):
                    with st.spinner("Registering user..."):
                        message, success = register_user(new_email, new_password)
                        st.toast(message, icon="✅" if success else "❌")
            
            st.markdown("---")
            st.markdown("<h4 style='text-align: center;'>Delete Users</h4>", unsafe_allow_html=True)
            all_users = get_all_users()
            if all_users:
                user_options = {f"{user['email']} (ID: {user['id']})": user['id'] for user in all_users}
                selected_user_ids = [user_options[option] for option in st.multiselect("Select users to delete:", list(user_options.keys()))]
                if st.button("**Delete Selected Users**", key="delete_users_button"):
                    if selected_user_ids: 
                        with st.spinner("Deleting users..."):
                            delete_users(selected_user_ids)
                    else: st.warning("Please select at least one user to delete.")
            else:
                st.info("No registered users to display.")

        with tab2:
            st.markdown("### Architecture Status")
            st.success("Powered by Google Cloud Vertex AI Vector Search.")
            st.markdown("---")
            
            upload_option = st.radio(
                "Choose an upload method:",
                ("Upload a File", "Add Text Content"),
                horizontal=True, key="upload_type"
            )

            if upload_option == "Upload a File":
                with st.form("file_upload_form", clear_on_submit=True):
                    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpeg", "jpg", "png", "txt", "csv", "xlsx", "docx"])
                    if st.form_submit_button("Upload and Index File"):
                        if uploaded_file:
                            original_bytes = uploaded_file.getvalue()
                            title = uploaded_file.name
                            ext = title.split('.')[-1].lower()
                            chunks, image_data = [], None

                            if ext in ["jpeg", "jpg", "png"]:
                                ocr_text = extract_text_from_image(io.BytesIO(original_bytes))
                                image_description = get_image_description(Image.open(io.BytesIO(original_bytes)))
                                combined_text = f"Image Description: {image_description}\n\nText found in image: {ocr_text}"
                                chunks = text_chunker(combined_text)
                                if combined_text.strip(): image_data = base64.b64encode(original_bytes).decode('utf-8')
                            elif ext == "pdf": chunks = text_chunker(extract_text_from_pdf(io.BytesIO(original_bytes)))
                            elif ext in ["txt", "csv"]: chunks = text_chunker(original_bytes.decode("utf-8"))
                            elif ext == "xlsx": chunks = get_structured_chunks_from_xlsx(io.BytesIO(original_bytes))
                            elif ext == "docx": chunks = text_chunker(extract_text_from_docx(io.BytesIO(original_bytes)))
                            
                            if chunks:
                                with st.spinner("Adding document to knowledge base..."):
                                    create_document_with_embedding(title, title, chunks, ext, image_data, original_file_bytes=original_bytes)
                        else:
                            st.warning("Please choose a file to upload.")

            elif upload_option == "Add Text Content":
                with st.form("text_form", clear_on_submit=True):
                    text_title = st.text_input("Title:")
                    text_content = st.text_area("Content:")
                    if st.form_submit_button("**Save Text**"):
                        if text_title and text_content:
                            filename = f"{text_title}.txt" if not text_title.lower().endswith('.txt') else text_title
                            chunks = text_chunker(text_content)
                            with st.spinner("Adding text to knowledge base..."):
                                create_document_with_embedding(text_title, filename, chunks, "string", original_file_bytes=text_content.encode('utf-8'))
                        else:
                            st.warning("Please provide both a title and content.")

            st.markdown("---")
            st.markdown("<h4 style='text-align: center;'>Delete Data</h4>", unsafe_allow_html=True)
            all_docs = get_all_documents()
            if all_docs:
                doc_options = {f"{doc['title']} (ID: {doc['id']})": (doc['id'], doc['filename']) for doc in all_docs}
                selected_options = st.multiselect("Select documents to delete:", list(doc_options.keys()))
                selected_docs_info = [doc_options[option] for option in selected_options]
                if st.button("**Delete Selected**", key="delete_docs_button"):
                    if selected_docs_info:
                        with st.spinner("Deleting selected documents..."):
                            delete_multiple_documents(selected_docs_info)
                    else:
                        st.warning("Please select at least one document.")
            else:
                st.info("No documents found.")

        if st.button("**Logout**", key="admin_logout_button"):
            with st.spinner("Logging out Admin..."):
                st.session_state.clear()
            st.rerun()

    else:
        st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                login_successful = False
                is_admin_attempt = check_admin(email, password)
                spinner_message = "Logging in as Admin..." if is_admin_attempt else f"Logging in as {email}..."

                with st.spinner(spinner_message):
                    if is_admin_attempt:
                        st.session_state.is_admin = True
                        st.session_state.user_id = "admin"
                        st.session_state.email = email
                        login_successful = True
                    else:
                        user_id, is_admin_user = check_user(email, password)
                        if user_id:
                            st.session_state.user_id = user_id
                            st.session_state.is_admin = is_admin_user
                            st.session_state.email = email
                            login_successful = True
                        else:
                            st.error("Invalid email or password.")
                
                if login_successful:
                    st.rerun()

# --- Main App Logic ---
logo_base_64 = get_image_as_base64(LOGO_PATH)
if logo_base_64:
    st.markdown(f"""<div style="display: flex; align-items: center; justify-content: center;"><img src="data:image/png;base64,{logo_base_64}" alt="Logo" style="height: 50px; margin-right: 15px;"><h1 style='color: #ff1493; margin: 0;'><b>Katrina Knowledgebase 🎂</b></h1></div>""", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align: center; color: #ff1493;'><b>Katrina Knowledgebase 🎂</b></h1>", unsafe_allow_html=True)
st.markdown("---")

if st.session_state.user_id:
    if st.session_state.is_admin:
        st.info("Welcome, Admin. Please use the sidebar to manage users and the knowledge base.")
    else:
        if 'welcome_message_shown' not in st.session_state:
            st.toast(f"Welcome to the Katrina Knowledgebase, {st.session_state.email}!", icon="🎂")
            st.session_state.welcome_message_shown = True

        _, chat_col, _ = st.columns([1, 2, 1])

        with chat_col:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # --- New, Simplified Workflow ---
            if 'search_query' not in st.session_state:
                st.session_state.search_query = None
            
            # --- Search Input Handling ---
            with st.form("image_upload_form", clear_on_submit=True):
                uploaded_image = st.file_uploader("Search with an image...", type=["jpeg", "jpg", "png"])
                submitted_image = st.form_submit_button("Search with Image")

            prompt = st.chat_input("Hi, I'm your Katrina Assistant, Ask me")

            if st.button("**New Search**", key="new_chat_main_button"):
                clear_search()
                st.rerun()
            
            # --- Process input and update state before the rest of the script runs ---
            if submitted_image and uploaded_image:
                img = Image.open(uploaded_image)
                st.session_state.search_query = get_image_description(img)
                st.session_state.search_image = img
                st.rerun()
            
            elif prompt:
                st.session_state.search_query = prompt
                st.session_state.search_image = None
                st.rerun()

            # --- Display Logic: Only runs if a search has been completed ---
            if st.session_state.search_query:
                search_query = st.session_state.search_query
                uploaded_image_obj = st.session_state.get('search_image')

                with st.spinner("Thinking..."):
                    intelligent_query = get_intelligent_search_query(search_query)
                    query_embedding = get_embedding(intelligent_query)
                    external_info = get_external_information(intelligent_query)
                    retrieved_docs = get_similar_documents(query_embedding)
                    if not retrieved_docs:
                        retrieved_docs = fallback_keyword_search(intelligent_query)
                    
                    suggestion = get_spelling_suggestion(search_query)
                    if suggestion.lower() == search_query.lower():
                        suggestion = None

                # Display the user's query and the assistant's response
                with st.chat_message("user"):
                    if uploaded_image_obj:
                        st.image(uploaded_image_obj, caption="User Upload")
                    st.markdown(search_query)
                
                with st.chat_message("assistant"):
                    if suggestion:
                         if st.button(f"Did you mean: **{suggestion}**?"):
                            st.session_state.search_query = suggestion
                            st.session_state.search_image = None
                            st.rerun()
                    
                    for doc in retrieved_docs:
                        with st.expander(f"Result: {doc.get('title', 'Untitled')}", expanded=True):
                            if 'image_data' in doc:
                                st.image(base64.b64decode(doc['image_data']), width=500)
                            if 'download_url' in doc and doc['download_url']:
                                st.markdown(f"[{doc['filename']}]({doc['download_url']}) ⬇️")
                    
                    response_stream = get_conversational_response_stream(search_query, retrieved_docs, external_info)
                    st.write_stream(response_stream)
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please log in using the sidebar to start.")

st.markdown('<div class="footer">Katrina Knowledgebase by Judy Sepe</div>', unsafe_allow_html=True)

