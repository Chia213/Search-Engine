
import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for ultra-modern styling
def load_css():
    st.markdown("""
    <style>
    /* Modern theme colors - 2024/2025 design trends */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --info-color: #3b82f6;
        --light-color: #f8fafc;
        --dark-color: #0f172a;
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
        
        /* Modern gradients */
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-modern: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        --gradient-dark: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        --gradient-card: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        
        /* Modern shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        
        /* Modern border radius */
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-2xl: 1.5rem;
        --radius-full: 9999px;
    }
    
    /* Global modern styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Ultra-modern header with glassmorphism */
    .main-header {
        background: var(--gradient-modern);
        padding: 3rem 2rem;
        border-radius: var(--radius-2xl);
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: var(--shadow-2xl);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--gradient-glass);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-2xl);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        position: relative;
        z-index: 1;
        font-weight: 500;
    }
    
    /* Ultra-modern card styling with glassmorphism */
    .search-card {
        background: var(--gradient-card);
        border-radius: var(--radius-2xl);
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-xl);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .search-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-modern);
        border-radius: var(--radius-2xl) var(--radius-2xl) 0 0;
    }
    
    .search-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-2xl);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Ultra-modern button styling */
    .stButton > button {
        background: var(--gradient-modern);
        color: white;
        border: none;
        border-radius: var(--radius-full);
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: var(--shadow-2xl);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* Ultra-modern input styling */
    .stTextInput > div > div > input {
        border-radius: var(--radius-full);
        border: 2px solid var(--gray-200);
        padding: 1rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: var(--gradient-card);
        box-shadow: var(--shadow-sm);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1), var(--shadow-lg);
        transform: scale(1.02);
    }
    
    /* Modern popular search buttons */
    .popular-search-btn {
        background: var(--gradient-modern);
        color: white;
        border: none;
        border-radius: var(--radius-full);
        padding: 0.6rem 1.5rem;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-md);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        display: inline-block;
    }
    
    .popular-search-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.6s;
    }
    
    .popular-search-btn:hover::before {
        left: 100%;
    }
    
    .popular-search-btn:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: var(--shadow-xl);
    }
    
    .popular-search-btn:active {
        transform: translateY(0) scale(1.02);
    }
    
    /* Ultra-modern sidebar styling */
    .css-1d391kg {
        background: var(--gradient-card);
    }
    
    .sidebar .sidebar-content {
        background: var(--gradient-card);
    }
    
    /* Modern metric cards with glassmorphism */
    .metric-card {
        background: var(--gradient-card);
        border-radius: var(--radius-xl);
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-modern);
        border-radius: var(--radius-xl) 0 0 var(--radius-xl);
    }
    
    .metric-card:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: var(--shadow-xl);
    }
    
    /* Ultra-modern result cards */
    .result-card {
        background: var(--gradient-card);
        border-radius: var(--radius-2xl);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-xl);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-modern);
        border-radius: var(--radius-2xl) var(--radius-2xl) 0 0;
    }
    
    .result-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: var(--shadow-2xl);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Tips section */
    .tips-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .tips-section h4 {
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Popular search buttons */
    .popular-search {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        display: inline-block;
    }
    
    .popular-search:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Image styling */
    .result-image {
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .result-image:hover {
        transform: scale(1.05);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #2ca02c 0%, #28a745 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #d62728 0%, #dc3545 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffc107 0%, #ffb300 100%);
        color: #212529;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)

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
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_embeddings_data():
    """Load pre-computed embeddings and metadata"""
    # Load embeddings
    image_embeddings = np.load('embeddings/image_embeddings.npy')
    text_embeddings = np.load('embeddings/text_embeddings.npy')
    
    # Load metadata
    metadata = pd.read_csv('embeddings/metadata.csv')
    
    # Load model info with error handling
    try:
        with open('embeddings/model_info.json', 'r') as f:
            model_info = json.load(f)
    except Exception as e:
        st.error(f"Error loading model info: {e}")
        # Fallback values
        model_info = {
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "num_images": 8091,
            "total_embeddings": 8091,
            "num_samples": 8091,
            "dataset": "Flickr8k",
            "processing_date": "2025-01-10",
            "device_used": "cpu"
        }

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

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="🔍 AI Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Search Engine</h1>
        <p>Powered by OpenAI CLIP • Multimodal Image & Text Search</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0; text-align: center;">⚙️ Search Options</h2>
    </div>
    """, unsafe_allow_html=True)

    # Search type selection with better styling
    st.sidebar.markdown("### 🔍 Search Type")
    search_type = st.sidebar.radio(
        "Choose your search method:",
        ["Text-to-Image Search", "Image-to-Text Search"],
        help="Select how you want to search through the dataset"
    )

    # Number of results with enhanced slider
    st.sidebar.markdown("### 📊 Results")
    top_k = st.sidebar.slider(
        "Number of results to display:",
        min_value=1,
        max_value=20,
        value=5,
        help="Adjust the number of top results you want to see"
    )

    # Ultra-modern popular searches section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔥 Popular Searches")
    st.sidebar.markdown("*Click any search to try it instantly*")

    popular_searches = [
        "dog playing", "children smiling", "red car", "food cooking",
        "person running", "cat sleeping", "blue sky", "water beach",
        "house building", "tree nature", "person walking", "animal pet"
    ]

    # Create modern popular search buttons with custom styling
    st.sidebar.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">
    """, unsafe_allow_html=True)
    
    for i, search in enumerate(popular_searches):
        if st.sidebar.button(f"🔍 {search}", key=f"popular_{i}", help=f"Search for: {search}"):
            st.session_state.popular_search = search
            st.session_state.auto_search = True
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Display dataset info with enhanced styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Information")
    
    # Get values and format properly with better fallbacks
    num_images = model_info.get('num_images', len(metadata) if len(metadata) > 0 else 'Unknown')
    num_embeddings = model_info.get('total_embeddings', model_info.get('num_samples', len(metadata) if len(metadata) > 0 else 'Unknown'))
    embedding_dim = model_info.get('embedding_dim', 512)
    model_name = model_info.get('model_name', 'openai/clip-vit-base-patch32')
    dataset = model_info.get('dataset', 'Flickr8k')
    processing_date = model_info.get('processing_date', '2025-01-10')

    # Format numbers properly
    images_text = f"{num_images:,}" if isinstance(num_images, int) else str(num_images)
    embeddings_text = f"{num_embeddings:,}" if isinstance(num_embeddings, int) else str(num_embeddings)
    model_display = model_name.split('/')[-1] if '/' in model_name else model_name
    
    # Add cache clearing button
    if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload data"):
        st.cache_data.clear()
        st.rerun()

    # Create metric cards with custom styling
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>📸 Total Images</strong><br>
        <span style="font-size: 1.5rem; color: #667eea;">{images_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>🧠 Total Embeddings</strong><br>
        <span style="font-size: 1.5rem; color: #667eea;">{embeddings_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>📐 Embedding Dimension</strong><br>
        <span style="font-size: 1.5rem; color: #667eea;">{embedding_dim}D</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>🤖 Model</strong><br>
        <span style="font-size: 1rem; color: #667eea;">{model_display}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>📁 Dataset</strong><br>
        <span style="font-size: 1rem; color: #667eea;">{dataset}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>📅 Processing Date</strong><br>
        <span style="font-size: 1rem; color: #667eea;">{processing_date}</span>
    </div>
    """, unsafe_allow_html=True)

    # Check if this is a demo dataset
    num_images = model_info.get('num_images', len(metadata))
    if isinstance(num_images, int) and num_images < 1000:
        st.warning(f"⚠️ **Demo Mode**: You're using a small subset ({num_images:,} images) of the full Flickr8k dataset. For production use, run the full dataset processing in Part 1 to get all 8,091 images.")

    # Main content area with enhanced styling
    if search_type == "Text-to-Image Search":
        st.markdown("""
        <div class="search-card">
            <h2 style="color: #667eea; margin-bottom: 1rem;">🔤 Text-to-Image Search</h2>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Enter a text description to find similar images using AI-powered semantic search</p>
        </div>
        """, unsafe_allow_html=True)

        # Ultra-modern search tips section with glassmorphism
        st.markdown("""
        <div style="background: var(--gradient-modern); 
                    padding: 2.5rem; 
                    border-radius: var(--radius-2xl); 
                    margin: 2rem 0;
                    box-shadow: var(--shadow-2xl);
                    position: relative;
                    overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                        background: var(--gradient-glass); 
                        backdrop-filter: blur(20px);
                        border-radius: var(--radius-2xl);"></div>
            <div style="position: relative; z-index: 1;">
                <h4 style="color: white; margin-bottom: 2rem; font-size: 1.5rem; font-weight: 700;">💡 Search Tips & Examples</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; align-items: start;">
                    <div style="background: rgba(255,255,255,0.1); 
                                padding: 1.5rem; 
                                border-radius: var(--radius-xl);
                                backdrop-filter: blur(10px);
                                border: 1px solid rgba(255,255,255,0.2);">
                        <strong style="color: white; font-size: 1.1rem;">🎯 Try searching for:</strong><br><br>
                        <div style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                            • <strong>Animals:</strong> 'dog', 'cat', 'bird'<br>
                            • <strong>Activities:</strong> 'playing', 'running', 'cooking'<br>
                            • <strong>Objects:</strong> 'car', 'house', 'food'<br>
                            • <strong>Emotions:</strong> 'smiling', 'happy', 'sad'
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); 
                                padding: 1.5rem; 
                                border-radius: var(--radius-xl);
                                backdrop-filter: blur(10px);
                                border: 1px solid rgba(255,255,255,0.2);">
                        <strong style="color: white; font-size: 1.1rem;">🚀 Quick Examples:</strong><br><br>
                        <div style="color: rgba(255,255,255,0.9); line-height: 1.8;">
                            Click any example below to search instantly!<br>
                            The AI will find the most relevant images.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Ultra-modern example query buttons
        st.markdown("### 🚀 Quick Search Examples")
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 1rem; margin: 2rem 0;">
        """, unsafe_allow_html=True)
        
        examples = [
            ("🐕 A dog playing", "Search for dogs playing", "a dog playing"),
            ("👶 Children smiling", "Search for smiling children", "children smiling"),
            ("🚗 Red car", "Search for red cars", "red car"),
            ("🍕 Food cooking", "Search for cooking food", "food cooking")
        ]
        
        for i, (text, help_text, query) in enumerate(examples):
            if st.button(text, key=f"example{i+1}", help=help_text):
                st.session_state.example_query = query
                st.session_state.auto_search = True
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced text input section
        st.markdown("### 🔍 Search Query")
        query_text = st.text_input(
            "Enter your search description:",
            placeholder="Describe what you're looking for... (e.g., 'a dog playing in the park', 'children smiling', 'red car on street')",
            help="💡 Be specific! Try describing objects, actions, colors, or emotions. The more descriptive, the better the results!",
            value=st.session_state.get('example_query', st.session_state.get('popular_search', '')),
            key="search_input",
            label_visibility="collapsed"
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

        # Enhanced search button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Search Images", type="primary", use_container_width=True) or should_search:
                if query_text:
                    with st.spinner("🔍 Searching for images..."):
                        results = text_to_image_search(query_text, top_k)

                    if results:
                        st.success(f"✅ Found {len(results)} results for: '{query_text}'")

                        # Display results with enhanced styling
                        st.markdown("### 🖼️ Search Results")
                        
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
                                        
                                        # Create result card with Streamlit components
                                        st.markdown(f"""
                                        <div class="result-card">
                                            <div style="text-align: center; padding: 1rem;">
                                                <h4 style="color: #667eea; margin: 0.5rem 0;">Similarity: {result['similarity']:.3f}</h4>
                                                <p style="color: #666; font-size: 0.9rem; margin: 0.25rem 0;">
                                                    <strong>ID:</strong> {result['image_id']}
                                                </p>
                                                <p style="color: #333; font-size: 0.95rem; margin: 0.5rem 0;">
                                                    {result['caption']}
                                                </p>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Display image using Streamlit
                                        st.image(image, caption=f"Similarity: {result['similarity']:.3f}", use_container_width=True)
                                    else:
                                        st.error(f"Image not found: {image_path}")
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                    else:
                        st.warning("⚠️ No results found. Try a different search query.")
                else:
                    st.warning("⚠️ Please enter a search query.")

    else:  # Image-to-Text Search
        st.markdown("""
        <div class="search-card">
            <h2 style="color: #667eea; margin-bottom: 1rem;">🖼️ Image-to-Text Search</h2>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">Upload an image to find similar text descriptions using AI-powered visual understanding</p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced upload guidance
        st.markdown("""
        <div class="tips-section">
            <h4>📋 Upload Guidelines</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div>
                    <strong>📁 Supported formats:</strong><br>
                    • JPG, JPEG<br>
                    • PNG<br>
                    • BMP, GIF
                </div>
                <div>
                    <strong>🎯 Best results with:</strong><br>
                    • Clear, well-lit images<br>
                    • Single main subject<br>
                    • Good contrast
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced image upload section
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="💡 Upload a clear image with a main subject for best search results!",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Display uploaded image with styling
            uploaded_image = Image.open(uploaded_file)
            st.markdown("### 📸 Your Uploaded Image")
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            # Enhanced search button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Search Descriptions", type="primary", use_container_width=True):
                    with st.spinner("🔍 Searching for similar descriptions..."):
                        results = image_to_text_search(uploaded_image, top_k)

                    if results:
                        st.success(f"✅ Found {len(results)} similar descriptions")

                        # Display results with enhanced styling
                        st.markdown("### 📝 Similar Descriptions")
                        
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
                                        
                                        # Create result card
                                        st.markdown(f"""
                                        <div class="result-card">
                                            <div style="text-align: center; padding: 1rem;">
                                                <h4 style="color: #667eea; margin: 0.5rem 0;">Similarity: {result['similarity']:.3f}</h4>
                                                <p style="color: #666; font-size: 0.9rem; margin: 0.25rem 0;">
                                                    <strong>ID:</strong> {result['image_id']}
                                                </p>
                                                <p style="color: #333; font-size: 0.95rem; margin: 0.5rem 0;">
                                                    {result['caption']}
                                                </p>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Display original image
                                        st.image(original_image, caption="Original Image", use_container_width=True)
                                    else:
                                        st.error(f"Original image not found: {image_path}")
                                except Exception as e:
                                    st.error(f"Error loading original image: {e}")
                    else:
                        st.warning("⚠️ No results found. Try a different image.")

if __name__ == "__main__":
    main()
