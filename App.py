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
from google.cloud import aiplatform
# BUG FIX: Use the direct, versioned import for IndexDatapoint
from google.cloud.aiplatform_v1.types import IndexDatapoint
import ast # Safely parse string representations of Python literals
import time # Import the time module for the sequential display effect

# --- UI Layout ---
LOGO_PATH = "Katrina_logo.png"
page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🎂"
st.set_page_config(page_title="Katrina Knowledgebase", page_icon=page_icon, layout="wide")

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

# --- Optimized Service Initialization ---
@st.cache_resource
def init_services():
    """
    Initializes and caches connections to all backend services. This function uses a robust
    authentication method that works for both local development and Streamlit Cloud deployment.
    """
    try:
        # --- Unified Credential Handling for Deployment ---
        cred_path = "gcp_creds.json"
        if os.path.exists("serviceAccountKey.json"):
            # Use local key file if it exists
            cred_path = "serviceAccountKey.json"
        elif "firebase" in st.secrets:
            # Otherwise, use secrets and write to a temporary file for cloud deployment
            firebase_creds = st.secrets["firebase"]
            if isinstance(firebase_creds, str):
                creds_dict = ast.literal_eval(firebase_creds)
            else:
                creds_dict = dict(firebase_creds)
            with open(cred_path, "w") as f:
                json.dump(creds_dict, f)
        else:
            st.error("Error: No Google Cloud credentials found.")
            st.stop()
        
        # Set the environment variable for Google Cloud libraries
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

        # --- Firestore Initialization ---
        if not firebase_admin._apps:
            cred_obj_for_firebase = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred_obj_for_firebase)
        db = fa_firestore.client()

        # --- Vertex AI Vector Search Initialization ---
        gcp_project_id = st.secrets["GCP_PROJECT_ID"]
        gcp_region = st.secrets["GCP_REGION"]
        vertex_ai_index_id = st.secrets["VERTEX_AI_INDEX_ID"]
        vertex_ai_endpoint_id = st.secrets["VERTEX_AI_ENDPOINT_ID"]
        if "VERTEX_AI_DEPLOYED_INDEX_ID" not in st.secrets:
            st.error("Configuration missing: Please add VERTEX_AI_DEPLOYED_INDEX_ID.")
            st.stop()

        full_index_name = f"projects/{gcp_project_id}/locations/{gcp_region}/indexes/{vertex_ai_index_id}"
        
        # Initialize without explicit credentials; it will use the environment variable.
        aiplatform.init(project=gcp_project_id, location=gcp_region)
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=vertex_ai_endpoint_id)

        # --- Gemini API and Embeddings Model Initialization ---
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=gemini_api_key)
        
    except Exception as e:
        st.error(f"Error during service initialization: {e}")
        st.stop()
        
    return db, index_endpoint, full_index_name

# Initialize all services at once and cache the result
db, index_endpoint, full_index_name = init_services()


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
        st.error("Error processing XLSX file: Missing optional dependency 'openpyxl'.")
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

def create_document_with_embedding(title, filename, content, content_type, image_data=None, original_file_bytes=None):
    doc_id = str(uuid.uuid4())
    doc_ref = db.collection("knowledge_base").document(doc_id)
    
    embedding = get_embedding(content)
    
    data_to_store = {
        "title": title, "filename": filename, "content": content,
        "content_type": content_type, "timestamp": fa_firestore.SERVER_TIMESTAMP,
        "uuid": doc_id
    }

    if image_data:
        image_bytes_decoded = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes_decoded))
        img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        data_to_store["image_data"] = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if original_file_bytes:
        data_to_store["original_file"] = base64.b64encode(original_file_bytes).decode('utf-8')
    
    doc_ref.set(data_to_store)

    try:
        index = aiplatform.MatchingEngineIndex(index_name=full_index_name)
        
        datapoint = IndexDatapoint(
            datapoint_id=doc_id,
            feature_vector=embedding
        )
        
        index.upsert_datapoints(datapoints=[datapoint])
        st.success(f"Successfully added '{title}' to the knowledge base and Vector Search index!")
        get_all_documents.clear()
    except Exception as e:
        st.error(f"Failed to add embedding to Vertex AI Vector Search: {e}")
        doc_ref.delete()
        st.warning(f"Removed '{title}' from Firestore due to indexing failure.")

def get_conversational_response_stream(user_query, retrieved_docs, external_info=None):
    """
    Generates a conversational response from the AI model as a stream.
    """
    if not retrieved_docs and not external_info:
        yield "I am sorry, but I cannot find any relevant information."
        return

    context_prepend = f"**Real-time Information from the Web:**\n{external_info}\n\n---\n\n" if external_info else ""
    
    context_list = []
    for doc in retrieved_docs:
        title = doc.get('title', 'Untitled Document')
        content = doc.get('content', '')
        prefix = f"The following is from the spreadsheet titled '{title}':\n" if doc.get('content_type') == 'xlsx' else f"The following is from the document titled '{title}':\n"
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
        # Use stream=True to get a generator object
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

        neighbor_ids = [neighbor.id for neighbor in response[0]]
        if not neighbor_ids:
            return []

        docs_ref = db.collection("knowledge_base")
        firestore_query = docs_ref.where("uuid", "in", neighbor_ids)
        return [doc.to_dict() for doc in firestore_query.stream()]
        
    except Exception as e:
        st.error(f"Error querying Vertex AI Vector Search: {e}")
        return []

def fallback_keyword_search(user_query):
    docs = db.collection("knowledge_base").stream()
    keywords = user_query.lower().split()
    results = [doc.to_dict() for doc in docs if any(keyword in doc.to_dict().get('content', '').lower() for keyword in keywords)]
    return results[:10]

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

def clear_chat_history():
    st.session_state.messages = []
    get_external_information.clear()
    get_intelligent_search_query.clear()

# UX IMPROVEMENT: Add a spinner to the document list refresh
@st.cache_data(show_spinner="Refreshing document list...")
def get_all_documents():
    docs = db.collection("knowledge_base").stream()
    return [{"id": doc.id, "title": doc.to_dict().get("title", "Untitled")} for doc in docs]

def delete_multiple_documents(doc_ids):
    try:
        for doc_id in doc_ids:
            db.collection("knowledge_base").document(doc_id).delete()
        
        index = aiplatform.MatchingEngineIndex(index_name=full_index_name)
        index.remove_datapoints(datapoint_ids=doc_ids)

        st.success(f"Successfully deleted {len(doc_ids)} documents.")
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
        st.info("You can now use the chat assistant.")
        if st.button("**Logout**", key="user_logout_button"):
            with st.spinner(f"Logging out {st.session_state.email}..."):
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
            st.markdown("<h4 style='text-align: center;'>Upload Files</h4>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpeg", "jpg", "png", "txt", "csv", "xlsx", "docx"], key="admin_uploader")
            if uploaded_file:
                original_bytes = uploaded_file.getvalue()
                title = uploaded_file.name
                ext = title.split('.')[-1].lower()
                content, image_data = None, None
                
                if ext == "pdf": content = extract_text_from_pdf(io.BytesIO(original_bytes))
                elif ext in ["jpeg", "jpg", "png"]:
                    content = extract_text_from_image(io.BytesIO(original_bytes))
                    if content: image_data = base64.b64encode(original_bytes).decode('utf-8')
                elif ext in ["txt", "csv"]: content = original_bytes.decode("utf-8")
                elif ext == "xlsx": content = extract_text_from_xlsx(io.BytesIO(original_bytes))
                elif ext == "docx": content = extract_text_from_docx(io.BytesIO(original_bytes))
                
                if content:
                    with st.expander("Show Extracted Content"):
                        st.text_area("Content:", value=content, height=300)
                    with st.spinner("Adding document to knowledge base..."):
                        create_document_with_embedding(title, title, content, ext, image_data, original_file_bytes=original_bytes)

            st.markdown("<h4 style='text-align: center;'>Add Text</h4>", unsafe_allow_html=True)
            with st.form("text_form", clear_on_submit=True):
                text_title = st.text_input("Title:")
                text_content = st.text_area("Content:")
                if st.form_submit_button("**Save Text**"):
                    if text_title and text_content:
                        filename = f"{text_title}.txt" if not text_title.lower().endswith('.txt') else text_title
                        with st.spinner("Adding text to knowledge base..."):
                            create_document_with_embedding(text_title, filename, text_content, "string", original_file_bytes=text_content.encode('utf-8'))
                    else: st.warning("Please provide both a title and content.")

            st.markdown("---")
            st.markdown("<h4 style='text-align: center;'>Delete Data</h4>", unsafe_allow_html=True)
            all_docs = get_all_documents()
            if all_docs:
                doc_options = {f"{doc['title']} (ID: {doc['id']})": doc['id'] for doc in all_docs}
                selected_doc_ids = [doc_options[option] for option in st.multiselect("Select documents to delete:", list(doc_options.keys()))]
                if st.button("**Delete Selected**", key="delete_docs_button"):
                    if selected_doc_ids:
                        with st.spinner("Deleting selected documents..."):
                            delete_multiple_documents(selected_doc_ids)
                    else: st.warning("Please select at least one document.")
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
                # BUG FIX: Check for admin credentials to customize the spinner message
                is_admin_attempt = check_admin(email, password)
                spinner_message = "Logging in as Admin..." if is_admin_attempt else f"Logging in as {email}..."

                with st.spinner(spinner_message):
                    if is_admin_attempt:
                        st.session_state.is_admin = True
                        st.session_state.user_id = "admin"
                        st.session_state.email = email
                    else:
                        user_id, is_admin_user = check_user(email, password)
                        if user_id:
                            st.session_state.user_id = user_id
                            st.session_state.is_admin = is_admin_user
                            st.session_state.email = email
                        else:
                            st.error("Invalid email or password.")
                    if "user_id" in st.session_state and st.session_state.user_id:
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
        # UX IMPROVEMENT: Add a welcome message toast for the user's first interaction.
        if 'welcome_message_shown' not in st.session_state:
            st.toast(f"Welcome to the Katrina Knowledgebase, {st.session_state.email}!", icon="🎂")
            st.session_state.welcome_message_shown = True

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Display retrieved documents if they exist
                for i, doc in enumerate(message.get("docs", [])):
                     with st.expander(f"Result: {doc.get('title', 'Untitled')}", expanded=True):
                        if 'image_data' in doc:
                            st.image(base64.b64decode(doc['image_data']), width=500)
                        if 'original_file' in doc and 'filename' in doc:
                            try:
                                file_bytes = base64.b64decode(doc['original_file'])
                                st.download_button(
                                    label=f"Download '{doc['filename']}'",
                                    data=file_bytes, file_name=doc['filename'],
                                    mime=get_mime_type(doc['filename']),
                                    key=f"dl_{i}_{doc['uuid']}_{message['timestamp']}" # Add timestamp for unique key
                                )
                            except Exception as e:
                                st.error(f"Could not prepare file for download: {e}")
                
                if 'image' in message: st.image(message['image'], caption="User Upload")

                # Display the text content of the message
                # If it was a streaming response, this will show the final, complete text.
                st.markdown(message["content"])

        # Refactored search logic to handle separate inputs cleanly
        def process_search(search_query, uploaded_image_obj=None):
            user_message = {}
            if uploaded_image_obj:
                user_message = {"role": "user", "content": f"Searched for image: {search_query}", "image": uploaded_image_obj, "timestamp": pd.Timestamp.now()}
            else: # Text prompt
                user_message = {"role": "user", "content": search_query, "timestamp": pd.Timestamp.now()}

            st.session_state.messages.append(user_message)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    intelligent_query = get_intelligent_search_query(search_query)
                    query_embedding = get_embedding(intelligent_query)
                    external_info = get_external_information(intelligent_query)
                    retrieved_docs = get_similar_documents(query_embedding)
                    if not retrieved_docs:
                        retrieved_docs = fallback_keyword_search(intelligent_query)

                # UX IMPROVEMENT: Sequentially display retrieved documents
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Result: {doc.get('title', 'Untitled')}", expanded=True):
                        if 'image_data' in doc:
                            st.image(base64.b64decode(doc['image_data']), width=500)
                        if 'original_file' in doc and 'filename' in doc:
                            try:
                                file_bytes = base64.b64decode(doc['original_file'])
                                st.download_button(
                                    label=f"Download '{doc['filename']}'",
                                    data=file_bytes, file_name=doc['filename'],
                                    mime=get_mime_type(doc['filename']),
                                    key=f"dl_new_{i}_{doc['uuid']}"
                                )
                            except Exception as e:
                                st.error(f"Could not prepare file for download: {e}")
                    time.sleep(0.5) # Add a delay for the sequential effect

                # Use st.write_stream to render the response as it comes in
                response_stream = get_conversational_response_stream(search_query, retrieved_docs, external_info)
                full_response = st.write_stream(response_stream)

            # Append the final, complete message to the session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "docs": retrieved_docs,
                "timestamp": pd.Timestamp.now()
            })
            st.rerun()

        # --- Search input area ---
        with st.form("image_upload_form"):
            uploaded_image = st.file_uploader("Search with an image...", type=["jpeg", "jpg", "png"])
            submitted = st.form_submit_button("Search with Image")
            if submitted and uploaded_image:
                img = Image.open(uploaded_image)
                search_query = get_image_description(img)
                process_search(search_query, uploaded_image_obj=img)

        prompt = st.chat_input("Hi, I'm your Katrina Assistant, Ask me")
        if prompt:
            process_search(prompt)
        
        # Place the "New Chat" button at the bottom of the main chat area
        if st.button("**New Chat**", key="new_chat_main_button"):
            clear_chat_history()
            st.rerun()

else:
    st.info("Please log in using the sidebar to start.")

