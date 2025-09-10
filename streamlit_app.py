

import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import io
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CLIP model
@st.cache_resource
def load_clip_model():
    """Load CLIP model and processor"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Load embeddings data
@st.cache_data
def load_embeddings_data():
    """Load pre-computed embeddings and metadata"""
    # Load embeddings
    image_embeddings = np.load('embeddings/image_embeddings.npy')
    text_embeddings = np.load('embeddings/text_embeddings.npy')

    # Load metadata
    metadata = pd.read_csv('embeddings/metadata.csv')

    # Load model info
    with open('embeddings/model_info.json', 'r') as f:
        model_info = json.load(f)

    return image_embeddings, text_embeddings, metadata, model_info

# Load model and data
model, processor = load_clip_model()
image_embeddings, text_embeddings, metadata, model_info = load_embeddings_data()

# Text-to-Image Search Function
def text_to_image_search(query_text, top_k=5):
    """Search for images based on text query"""
    # Generate embedding for text query
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    # Calculate similarities with all image embeddings
    similarities = cosine_similarity(query_embedding.cpu().numpy(), image_embeddings)[0]

    # Get top-k most similar images
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = {
            'image_id': metadata.iloc[idx]['image_id'],
            'image_path': metadata.iloc[idx]['image_path'],
            'caption': metadata.iloc[idx]['caption'],
            'similarity': similarities[idx]
        }
        results.append(result)

    return results

# Image-to-Text Search Function
def image_to_text_search(uploaded_image, top_k=5):
    """Search for text descriptions based on uploaded image"""
    # Generate embedding for uploaded image
    inputs = processor(images=uploaded_image, return_tensors="pt").to(device)

    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    # Calculate similarities with all text embeddings
    similarities = cosine_similarity(query_embedding.cpu().numpy(), text_embeddings)[0]

    # Get top-k most similar text descriptions
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = {
            'image_id': metadata.iloc[idx]['image_id'],
            'image_path': metadata.iloc[idx]['image_path'],
            'caption': metadata.iloc[idx]['caption'],
            'similarity': similarities[idx]
        }
        results.append(result)

    return results

# Custom CSS for modern UI
def load_css():
    st.markdown("""
    <style>
    /* Modern theme colors - Clean & Minimal */
    :root {
        --primary-color: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary-color: #7c3aed;
        --accent-color: #06b6d4;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --dark-color: #111827;
        --dark-light: #374151;
        --light-color: #ffffff;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-400: #9ca3af;
        --gray-500: #6b7280;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
        --gradient-primary: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #7c3aed 100%);
        --gradient-secondary: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        --gradient-accent: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        --border-radius: 8px;
        --border-radius-md: 12px;
        --border-radius-lg: 16px;
        --border-radius-xl: 20px;
        --border-radius-2xl: 24px;
    }

    /* Reset and base styles */
    * {
        box-sizing: border-box;
    }

    /* Main container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
        background: var(--gray-50);
        min-height: 100vh;
    }

    /* Header styling - Clean & Modern */
    .main-header {
        background: var(--light-color);
        padding: 3rem 2rem;
        border-radius: var(--border-radius-2xl);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        text-align: center;
        color: var(--dark-color);
        position: relative;
        overflow: hidden;
        border: 1px solid var(--gray-200);
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        color: var(--dark-color);
        position: relative;
        z-index: 1;
        letter-spacing: -0.025em;
    }

    .main-header p {
        font-size: 1.125rem;
        margin: 1rem 0 0 0;
        color: var(--gray-600);
        position: relative;
        z-index: 1;
        font-weight: 400;
    }

    /* Hero section - Clean stats */
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
        flex-wrap: wrap;
    }

    .hero-stat {
        text-align: center;
        background: var(--gray-50);
        padding: 1.5rem 1.25rem;
        border-radius: var(--border-radius-lg);
        border: 1px solid var(--gray-200);
        min-width: 120px;
        transition: all 0.2s ease;
    }

    .hero-stat:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }

    .hero-stat .number {
        font-size: 1.75rem;
        font-weight: 700;
        display: block;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }

    .hero-stat .label {
        font-size: 0.875rem;
        color: var(--gray-600);
        margin: 0;
        font-weight: 500;
    }

    /* Sidebar styling - Clean & Minimal */
    .css-1d391kg {
        background: var(--light-color);
        border-right: 1px solid var(--gray-200);
    }

    .sidebar .sidebar-content {
        background: var(--light-color);
        padding: 1.5rem 1rem;
    }

    .sidebar .sidebar-content .element-container {
        margin-bottom: 1.5rem;
    }

    .sidebar h3 {
        color: var(--dark-color);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--gray-200);
        letter-spacing: 0.025em;
    }

    /* Card styling - Clean & Modern */
    .metric-card {
        background: var(--light-color);
        padding: 1.25rem;
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--gray-200);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-primary);
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }

    .metric-card h3 {
        color: var(--gray-600);
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--dark-color);
        margin: 0;
    }

    /* Search card - Clean & Modern */
    .search-card {
        background: var(--light-color);
        border-radius: var(--border-radius-xl);
        padding: 2rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--gray-200);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .search-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }

    /* Results grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }

    /* Button styling - Clean & Modern */
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: var(--border-radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }

    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: var(--primary-color);
        box-shadow: var(--shadow-md);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--primary-dark);
        box-shadow: var(--shadow-lg);
    }

    /* Search input styling - Clean & Modern */
    .stTextInput > div > div > input {
        border-radius: var(--border-radius-md);
        border: 1px solid var(--gray-300);
        padding: 0.875rem 1rem;
        font-size: 1rem;
        font-weight: 400;
        transition: all 0.2s ease;
        background: var(--light-color);
        box-shadow: var(--shadow-xs);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1), var(--shadow-xs);
        background: var(--light-color);
        outline: none;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--gray-400);
        font-weight: 400;
    }

    /* Popular search buttons - Clean & Modern */
    .popular-search-btn {
        background: var(--light-color);
        border: 1px solid var(--gray-300);
        border-radius: var(--border-radius);
        padding: 0.5rem 0.875rem;
        margin: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s ease;
        display: inline-block;
        text-decoration: none;
        color: var(--gray-700);
        box-shadow: var(--shadow-xs);
    }

    .popular-search-btn:hover {
        border-color: var(--primary-color);
        background: var(--primary-color);
        color: white;
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
    }

    /* Results styling - Clean & Modern */
    .result-card {
        background: var(--light-color);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-sm);
        overflow: hidden;
        transition: all 0.2s ease;
        border: 1px solid var(--gray-200);
    }

    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
    }

    .result-card img {
        width: 100%;
        height: 200px;
        object-fit: cover;
    }

    .result-card .content {
        padding: 1rem;
    }

    .result-card .similarity {
        background: var(--primary-color);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: var(--border-radius);
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    .result-card .caption {
        color: var(--gray-700);
        font-size: 0.875rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Status messages - Clean & Modern */
    .stSuccess {
        background: var(--success-color);
        color: white;
        padding: 1rem;
        border-radius: var(--border-radius-md);
        border: none;
        box-shadow: var(--shadow-sm);
    }

    .stError {
        background: var(--error-color);
        color: white;
        padding: 1rem;
        border-radius: var(--border-radius-md);
        border: none;
        box-shadow: var(--shadow-sm);
    }

    .stWarning {
        background: var(--warning-color);
        color: white;
        padding: 1rem;
        border-radius: var(--border-radius-md);
        border: none;
        box-shadow: var(--shadow-sm);
    }

    /* Tabs styling - Clean & Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: var(--gray-100);
        padding: 0.25rem;
        border-radius: var(--border-radius-lg);
        margin-bottom: 2rem;
        border: 1px solid var(--gray-200);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--border-radius-md);
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        color: var(--gray-600);
        position: relative;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--gray-200);
        color: var(--primary-color);
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-color);
        color: white;
        box-shadow: var(--shadow-sm);
    }

    /* Slider styling - Clean & Modern */
    .stSlider > div > div > div > div {
        background: var(--primary-color);
    }

    /* File uploader styling - Clean & Modern */
    .stFileUploader > div > div > div {
        border: 2px dashed var(--gray-300);
        border-radius: var(--border-radius-lg);
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
        background: var(--gray-50);
    }

    .stFileUploader > div > div > div:hover {
        border-color: var(--primary-color);
        background: rgba(37, 99, 235, 0.05);
    }

    /* Responsive design - Clean & Modern */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }

        .main-header {
            padding: 2rem 1rem;
        }

        .main-header h1 {
            font-size: 2rem;
        }

        .main-header p {
            font-size: 1rem;
        }

        .hero-stats {
            gap: 1rem;
        }

        .hero-stat {
            min-width: 100px;
            padding: 1rem 0.75rem;
        }

        .hero-stat .number {
            font-size: 1.5rem;
        }

        .hero-stat .label {
            font-size: 0.75rem;
        }
    }

    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.75rem;
        }

        .hero-stats {
            flex-direction: column;
            align-items: center;
        }

        .hero-stat {
            width: 100%;
            max-width: 200px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="🔍 Multimodal Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clear cache if needed (for debugging)
    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Load custom CSS
    load_css()

    # Modern header with hero stats
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Multimodal Search Engine</h1>
        <p>Powered by OpenAI CLIP • Find images with text or text with images</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="number">""" + str(model_info.get('num_images', 100)) + """</span>
                <span class="label">Images</span>
            </div>
            <div class="hero-stat">
                <span class="number">""" + str(model_info.get('embedding_dim', 512)) + """D</span>
                <span class="label">Embeddings</span>
            </div>
            <div class="hero-stat">
                <span class="number">""" + str(model_info.get('dataset', 'Flickr8k')) + """</span>
                <span class="label">Dataset</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Modern sidebar
    st.sidebar.markdown("### ⚙️ Search Configuration")

    # Search type selection with modern styling
    search_type = st.sidebar.selectbox(
        "🔍 Search Type",
        ["Text-to-Image Search", "Image-to-Text Search"],
        help="Choose how you want to search"
    )

    # Number of results with modern slider
    st.sidebar.markdown("### 📊 Results")
    top_k = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top results to display"
    )

    # Popular searches with modern grid
    st.sidebar.markdown("### 🔥 Popular Searches")
    st.sidebar.markdown("*Click any suggestion to search instantly*")

    popular_searches = [
        "dog playing", "children smiling", "red car", "food cooking",
        "person running", "cat sleeping", "blue sky", "water beach",
        "house building", "tree nature", "person walking", "animal pet"
    ]

    # Create a grid of popular search buttons
    cols = st.sidebar.columns(2)
    for i, search in enumerate(popular_searches):
        with cols[i % 2]:
            if st.button(f"🔍 {search}", key=f"popular_{i}", help=f"Search for '{search}'"):
                st.session_state.popular_search = search
                st.session_state.auto_search = True

    # Dataset information with modern cards
    st.sidebar.markdown("### 📊 Dataset Information")

    # Get values and format properly
    num_images = model_info.get('num_images', 100)
    num_embeddings = model_info.get('total_embeddings', model_info.get('num_samples', 500))
    embedding_dim = model_info.get('embedding_dim', 512)
    model_name = model_info.get('model_name', 'CLIP Model')
    dataset = model_info.get('dataset', 'Flickr8k')
    processing_date = model_info.get('processing_date', datetime.now().strftime('%Y-%m-%d'))

    # Format numbers properly
    images_text = f"{num_images:,}" if isinstance(num_images, int) else str(num_images)
    embeddings_text = f"{num_embeddings:,}" if isinstance(num_embeddings, int) else str(num_embeddings)
    model_display = model_name.split('/')[-1] if '/' in model_name else model_name

    # Display metrics in modern cards
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>📸 Total Images</h3>
        <div class="value">{images_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>🔢 Total Embeddings</h3>
        <div class="value">{embeddings_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>📐 Embedding Dimension</h3>
        <div class="value">{embedding_dim}D</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>🤖 Model</h3>
        <div class="value">{model_display}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>📁 Dataset</h3>
        <div class="value">{dataset}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>📅 Processing Date</h3>
        <div class="value">{processing_date}</div>
    </div>
    """, unsafe_allow_html=True)

    # Check if this is a demo dataset
    num_images = model_info.get('num_images', len(metadata))
    if isinstance(num_images, int) and num_images < 1000:
        st.warning(f"⚠️ **Demo Mode**: You're using a small subset ({num_images:,} images) of the full Flickr8k dataset. For production use, run the full dataset processing in Part 1 to get all 8,091 images.")

    # Main content area with modern tabs
    tab1, tab2 = st.tabs(["🔤 Text-to-Image Search", "🖼️ Image-to-Text Search"])

    # Add clarification about the tabs
    st.info("💡 **Tip**: Use the **Text-to-Image** tab to search for images using text descriptions. Use the **Image-to-Text** tab to upload an image and find similar text descriptions.")

    with tab1:
        st.markdown("""
        <div class="search-card">
            <h2 style="margin: 0 0 1rem 0; color: var(--dark-color); font-size: 1.5rem; font-weight: 700;">🔤 Text-to-Image Search</h2>
            <p style="margin: 0 0 2rem 0; color: var(--gray-600); font-size: 1rem;">Describe what you're looking for and discover relevant images from the dataset</p>
        </div>
        """, unsafe_allow_html=True)

        # Search suggestions with modern cards
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div style="background: var(--gray-50); padding: 1.5rem; border-radius: var(--border-radius-lg); border-left: 3px solid var(--primary-color); margin-bottom: 1rem;">
                <h4 style="margin: 0 0 1rem 0; color: var(--dark-color); font-size: 1rem; font-weight: 600;">💡 Search Tips</h4>
                <div style="color: var(--gray-600); line-height: 1.6; font-size: 0.875rem;">
                    <strong>Try searching for:</strong><br>
                    • <strong>Animals:</strong> 'dog', 'cat', 'bird', 'horse'<br>
                    • <strong>Activities:</strong> 'playing', 'running', 'cooking'<br>
                    • <strong>Objects:</strong> 'car', 'house', 'food'<br>
                    • <strong>Emotions:</strong> 'smiling', 'happy', 'sad'<br>
                    • <strong>Scenes:</strong> 'beach', 'park', 'kitchen'
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: var(--gray-50); padding: 1.5rem; border-radius: var(--border-radius-lg); border-left: 3px solid var(--success-color); margin-bottom: 1rem;">
                <h4 style="margin: 0 0 1rem 0; color: var(--dark-color); font-size: 1rem; font-weight: 600;">🎯 Quick Examples</h4>
                <div style="color: var(--gray-600); line-height: 1.6; font-size: 0.875rem;">
                    Click any example to search instantly:
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Example buttons in a grid
            example_cols = st.columns(2)
            with example_cols[0]:
                if st.button("🐕 A dog playing", key="example1", help="Search for 'a dog playing'"):
                    st.session_state.example_query = "a dog playing"
                    st.session_state.auto_search = True
                if st.button("👶 Children smiling", key="example2", help="Search for 'children smiling'"):
                    st.session_state.example_query = "children smiling"
                    st.session_state.auto_search = True

            with example_cols[1]:
                if st.button("🚗 Red car", key="example3", help="Search for 'red car'"):
                    st.session_state.example_query = "red car"
                    st.session_state.auto_search = True
                if st.button("🍕 Food cooking", key="example4", help="Search for 'food cooking'"):
                    st.session_state.example_query = "food cooking"
                    st.session_state.auto_search = True

        # Text input with better placeholder
        query_text = st.text_input(
            "🔍 Enter your search query:",
            placeholder="Describe what you're looking for... (e.g., 'a dog playing in the park', 'children smiling', 'red car on street')",
            help="💡 Be specific! Try describing objects, actions, colors, or emotions. The more descriptive, the better the results!",
            value=st.session_state.get('example_query', st.session_state.get('popular_search', '')),
            key="search_input"
        )

        # Clear example queries after use
        if 'example_query' in st.session_state:
            del st.session_state.example_query
        if 'popular_search' in st.session_state:
            del st.session_state.popular_search

        # Check if we should auto-search (from popular searches or example queries)
        should_search = st.session_state.get('auto_search', False)
        if should_search:
            st.session_state.auto_search = False  # Reset the flag
            # Use example query if available, otherwise use popular search
            query_text = st.session_state.get('example_query', st.session_state.get('popular_search', query_text))

        if st.button("🔍 Search Images", type="primary") or should_search:
            if query_text:
                with st.spinner("Searching for images..."):
                    results = text_to_image_search(query_text, top_k)

                if results:
                    st.success(f"Found {len(results)} results for: '{query_text}'")

                    # Display results in columns
                    cols = st.columns(min(3, len(results)))
                    for i, result in enumerate(results):
                        with cols[i % 3]:
                            try:
                                image_path = result['image_path']
                                # Fix path - remove ../ if present
                                if image_path.startswith('../'):
                                    image_path = image_path[3:]  # Remove ../

                                if os.path.exists(image_path):
                                    image = Image.open(image_path)
                                    st.image(image, caption=f"Similarity: {result['similarity']:.3f}", use_container_width=True)

                                    # Display details
                                    st.markdown(f"**Image ID:** {result['image_id']}")
                                    st.markdown(f"**Caption:** {result['caption']}")
                                    st.markdown(f"**Similarity:** {result['similarity']:.3f}")
                                else:
                                    st.error(f"Image not found: {image_path}")
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                    else:
                        st.warning("No results found. Try a different search query.")
            else:
                st.warning("Please enter a search query.")

    with tab2:
        st.markdown("""
        <div class="search-card">
            <h2 style="margin: 0 0 1rem 0; color: var(--dark-color); font-size: 1.5rem; font-weight: 700;">🖼️ Image-to-Text Search</h2>
            <p style="margin: 0 0 2rem 0; color: var(--gray-600); font-size: 1rem;">Upload an image to find similar text descriptions from the dataset</p>
        </div>
        """, unsafe_allow_html=True)

        # Upload guidance
        st.markdown("#### 📋 Upload Guidelines")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: var(--gray-50); padding: 1.25rem; border-radius: var(--border-radius-lg); border-left: 3px solid var(--primary-color);">
                <h4 style="margin: 0 0 0.75rem 0; color: var(--dark-color); font-size: 0.875rem; font-weight: 600;">Supported formats:</h4>
                <div style="color: var(--gray-600); font-size: 0.875rem; line-height: 1.5;">
                    • JPG, JPEG<br>
                    • PNG<br>
                    • BMP, GIF
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: var(--gray-50); padding: 1.25rem; border-radius: var(--border-radius-lg); border-left: 3px solid var(--success-color);">
                <h4 style="margin: 0 0 0.75rem 0; color: var(--dark-color); font-size: 0.875rem; font-weight: 600;">Best results with:</h4>
                <div style="color: var(--gray-600); font-size: 0.875rem; line-height: 1.5;">
                    • Clear, well-lit images<br>
                    • Single main subject<br>
                    • Good contrast
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Image upload
        uploaded_file = st.file_uploader(
            "📁 Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="💡 Upload a clear image with a main subject for best search results!",
            label_visibility="collapsed"
        )

        # Add some guidance
        if not uploaded_file:
            st.info("👆 **Upload an image above** to find similar text descriptions from the dataset")

        if uploaded_file is not None:
            try:
                # Debug information
                st.write(f"📁 File name: {uploaded_file.name}")
                st.write(f"📏 File size: {uploaded_file.size} bytes")
                st.write(f"🔍 File type: {uploaded_file.type}")

                # Reset file pointer to beginning
                uploaded_file.seek(0)

                # Try using BytesIO with proper handling
                file_bytes = uploaded_file.read()
                st.write(f"📊 File bytes length: {len(file_bytes)}")

                # Check if file has content
                if len(file_bytes) == 0:
                    st.error("❌ File is empty!")
                    return

                # Try to create image from bytes using a more robust approach
                try:
                    # Create BytesIO object
                    image_io = io.BytesIO(file_bytes)
                    image_io.seek(0)

                    # Try to determine format from file extension
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    st.write(f"🔍 Detected file extension: {file_extension}")

                    # Try to open with PIL - let it auto-detect the format
                    uploaded_image = Image.open(image_io)

                    # Load the image data
                    uploaded_image.load()

                    # Convert to RGB if necessary
                    if uploaded_image.mode != 'RGB':
                        uploaded_image = uploaded_image.convert('RGB')

                    st.success("✅ Image loaded successfully!")

                except Exception as img_error:
                    st.error(f"❌ Error loading image: {str(img_error)}")

                    # Try alternative approach - save to temporary file
                    st.write("🔄 Trying temporary file approach...")

                    try:
                        import tempfile

                        # Create temporary file with proper extension
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                            tmp_file.write(file_bytes)
                            tmp_file_path = tmp_file.name

                        st.write(f"📁 Created temp file: {tmp_file_path}")

                        # Load from temporary file
                        uploaded_image = Image.open(tmp_file_path)
                        uploaded_image.load()

                        # Convert to RGB if necessary
                        if uploaded_image.mode != 'RGB':
                            uploaded_image = uploaded_image.convert('RGB')

                        # Clean up temporary file
                        os.unlink(tmp_file_path)

                        st.success("✅ Image loaded with temporary file method!")

                    except Exception as temp_error:
                        st.error(f"❌ Temporary file method failed: {str(temp_error)}")

                        # Final fallback - try with cv2 if available
                        st.write("🔄 Trying OpenCV fallback...")
                        try:
                            import cv2
                            import numpy as np

                            # Convert bytes to numpy array
                            nparr = np.frombuffer(file_bytes, np.uint8)

                            # Decode image with OpenCV
                            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if cv_image is not None:
                                # Convert BGR to RGB
                                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

                                # Convert to PIL Image
                                uploaded_image = Image.fromarray(cv_image)

                                st.success("✅ Image loaded with OpenCV fallback!")
                            else:
                                raise Exception("OpenCV could not decode the image")

                        except ImportError:
                            st.error("❌ OpenCV not available for fallback")
                            st.warning("The uploaded file might be corrupted or in an unsupported format.")
                            return
                        except Exception as cv_error:
                            st.error(f"❌ OpenCV fallback failed: {str(cv_error)}")
                            st.warning("The uploaded file might be corrupted or in an unsupported format.")
                            return

                # Display the image
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Search Descriptions", type="primary"):
                    with st.spinner("Searching for similar descriptions..."):
                        try:
                            # Use the image we already loaded
                            results = image_to_text_search(uploaded_image, top_k)
                        except Exception as search_error:
                            st.error(f"❌ Error during search: {str(search_error)}")
                            results = []

                    if results:
                        st.success(f"Found {len(results)} similar descriptions")

                        # Display results in columns
                        cols = st.columns(min(3, len(results)))
                        for i, result in enumerate(results):
                            with cols[i % 3]:
                                try:
                                    image_path = result['image_path']
                                    # Fix path - remove ../ if present
                                    if image_path.startswith('../'):
                                        image_path = image_path[3:]  # Remove ../

                                    if os.path.exists(image_path):
                                        original_image = Image.open(image_path)
                                        st.image(original_image, caption="Original Image", use_container_width=True)
                                    else:
                                        st.error(f"Original image not found: {image_path}")
                                except Exception as e:
                                    st.error(f"Error loading original image: {e}")

                                # Display details
                                st.markdown(f"**Image ID:** {result['image_id']}")
                                st.markdown(f"**Caption:** {result['caption']}")
                                st.markdown(f"**Similarity:** {result['similarity']:.3f}")
                    else:
                        st.warning("No results found. Try a different image.")

            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.warning("Please make sure you're uploading a valid image file (JPG, PNG, BMP, GIF)")
                uploaded_file = None

if __name__ == "__main__":
    main()


