
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

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="🔍 Multimodal Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .search-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .search-tips {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Multimodal Search Engine</h1>
        <p style="font-size: 1.2rem; margin: 0;">Powered by OpenAI CLIP • Find images with text or text with images</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with better styling
    st.sidebar.markdown("### ⚙️ Search Configuration")
    
    # Search type selection with better styling
    search_type = st.sidebar.selectbox(
        "🔍 Search Type:",
        ["Text-to-Image Search", "Image-to-Text Search"],
        help="Choose how you want to search"
    )

    # Number of results with better styling
    st.sidebar.markdown("### 📊 Results Settings")
    top_k = st.sidebar.slider(
        "Number of results:",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top results to display"
    )

    # Popular searches with better organization
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔥 Quick Searches")
    st.sidebar.markdown("*Click any suggestion to search instantly*")

    # Organize popular searches by category
    categories = {
        "🐕 Animals": ["dog playing", "cat sleeping", "bird flying", "animal pet"],
        "👥 People": ["children smiling", "person running", "person walking", "people talking"],
        "🚗 Objects": ["red car", "house building", "food cooking", "blue sky"],
        "🌊 Nature": ["water beach", "tree nature", "mountain view", "sunset sky"]
    }

    for category, searches in categories.items():
        st.sidebar.markdown(f"**{category}**")
        for i, search in enumerate(searches):
            if st.sidebar.button(f"• {search}", key=f"popular_{category}_{i}", help=f"Search for: {search}"):
                st.session_state.popular_search = search
                st.session_state.auto_search = True
        st.sidebar.markdown("")

    # Display dataset info with better styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Information")

    # Get values and format properly
    num_images = model_info.get('num_images', 'Unknown')
    num_embeddings = model_info.get('total_embeddings', model_info.get('num_samples', 'Unknown'))
    embedding_dim = model_info.get('embedding_dim', 'Unknown')
    model_name = model_info.get('model_name', 'Unknown')
    dataset = model_info.get('dataset', 'Unknown')
    processing_date = model_info.get('processing_date', datetime.now().strftime('%Y-%m-%d'))

    # Format numbers properly
    images_text = f"{num_images:,}" if isinstance(num_images, int) else str(num_images)
    embeddings_text = f"{num_embeddings:,}" if isinstance(num_embeddings, int) else str(num_embeddings)
    model_display = model_name.split('/')[-1] if '/' in model_name else model_name

    # Create metric cards
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h4>📸 Total Images</h4>
        <h2>{images_text}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h4>🧠 Embeddings</h4>
        <h2>{embeddings_text}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h4>⚡ Model</h4>
        <h3>{model_display}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h4>📚 Dataset</h4>
        <h3>{dataset}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Check if this is a demo dataset
    num_images = model_info.get('num_images', len(metadata))
    if isinstance(num_images, int) and num_images < 1000:
        st.warning(f"⚠️ **Demo Mode**: You're using a small subset ({num_images:,} images) of the full Flickr8k dataset. For production use, run the full dataset processing in Part 1 to get all 8,091 images.")

    # Main content area with better styling
    if search_type == "Text-to-Image Search":
        # Create a search card
        st.markdown("""
        <div class="search-card">
            <h2>🔤 Text-to-Image Search</h2>
            <p>Describe what you're looking for and find similar images instantly!</p>
        </div>
        """, unsafe_allow_html=True)

        # Search tips with better styling
        st.markdown("""
        <div class="search-tips">
            <h4>💡 Search Tips & Examples</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Search Categories:**")
            st.markdown("• **Animals:** 'dog playing', 'cat sleeping', 'bird flying'")
            st.markdown("• **Activities:** 'person running', 'children playing', 'cooking food'")
            st.markdown("• **Objects:** 'red car', 'blue house', 'delicious food'")
            st.markdown("• **Emotions:** 'happy people', 'smiling children', 'sad person'")

        with col2:
            st.markdown("**🚀 Try These Examples:**")
            example_buttons = [
                ("🐕", "A dog playing", "example1"),
                ("👶", "Children smiling", "example2"),
                ("🚗", "Red car", "example3"),
                ("🍕", "Food cooking", "example4"),
                ("🏃", "Person running", "example5"),
                ("🏠", "House building", "example6")
            ]
            
            for emoji, text, key in example_buttons:
                if st.button(f"{emoji} {text}", key=key, help=f"Search for: {text}"):
                    st.session_state.example_query = text
                    st.session_state.auto_search = True

        # Text input with better styling
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

        # Search button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Search Images", type="primary", use_container_width=True) or should_search:
                if query_text:
                    with st.spinner("🔍 Searching for images..."):
                        results = text_to_image_search(query_text, top_k)

                    if results:
                        st.success(f"✅ Found {len(results)} results for: '{query_text}'")

                        # Display results with better styling
                        st.markdown("### 🖼️ Search Results")
                        
                        # Create a grid layout for results
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
                                        
                                        # Create a result card
                                        st.markdown(f"""
                                        <div class="result-card">
                                            <h4>Rank #{i+1} (Similarity: {result['similarity']:.3f})</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.image(image, use_container_width=True)
                                        
                                        # Display details in a clean format
                                        st.markdown(f"**📝 Caption:** {result['caption']}")
                                        st.markdown(f"**🆔 Image ID:** {result['image_id']}")
                                        st.markdown(f"**📊 Similarity Score:** {result['similarity']:.3f}")
                                        st.markdown("---")
                                    else:
                                        st.error(f"❌ Image not found: {image_path}")
                                except Exception as e:
                                    st.error(f"❌ Error loading image: {e}")
                    else:
                        st.warning("⚠️ No results found. Try a different search query.")
                else:
                    st.warning("⚠️ Please enter a search query.")

    else:  # Image-to-Text Search
        # Create a search card for image search
        st.markdown("""
        <div class="search-card">
            <h2>🖼️ Image-to-Text Search</h2>
            <p>Upload an image to find similar text descriptions and related images!</p>
        </div>
        """, unsafe_allow_html=True)

        # Upload guidance with better styling
        st.markdown("""
        <div class="search-tips">
            <h4>📋 Upload Guidelines</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📁 Supported Formats:**")
            st.markdown("• **JPG, JPEG** - Most common")
            st.markdown("• **PNG** - High quality")
            st.markdown("• **BMP, GIF** - Also supported")

        with col2:
            st.markdown("**✨ Best Results With:**")
            st.markdown("• **Clear, well-lit images**")
            st.markdown("• **Single main subject**")
            st.markdown("• **Good contrast & focus**")
            st.markdown("• **High resolution**")

        # Image upload with better styling
        st.markdown("### 📁 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="💡 Upload a clear image with a main subject for best search results!",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Display uploaded image with better styling
            uploaded_image = Image.open(uploaded_file)
            st.markdown("### 📸 Your Uploaded Image")
            st.image(uploaded_image, caption="Your uploaded image", use_container_width=True)

            # Search button with better styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Search Descriptions", type="primary", use_container_width=True):
                    with st.spinner("🔍 Searching for similar descriptions..."):
                        results = image_to_text_search(uploaded_image, top_k)

                    if results:
                        st.success(f"✅ Found {len(results)} similar descriptions")

                        # Display results with better styling
                        st.markdown("### 🖼️ Similar Images & Descriptions")
                        
                        # Create a grid layout for results
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
                                        
                                        # Create a result card
                                        st.markdown(f"""
                                        <div class="result-card">
                                            <h4>Rank #{i+1} (Similarity: {result['similarity']:.3f})</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.image(original_image, use_container_width=True)
                                        
                                        # Display details in a clean format
                                        st.markdown(f"**📝 Caption:** {result['caption']}")
                                        st.markdown(f"**🆔 Image ID:** {result['image_id']}")
                                        st.markdown(f"**📊 Similarity Score:** {result['similarity']:.3f}")
                                        st.markdown("---")
                                    else:
                                        st.error(f"❌ Original image not found: {image_path}")
                                except Exception as e:
                                    st.error(f"❌ Error loading original image: {e}")
                    else:
                        st.warning("⚠️ No results found. Try a different image.")

if __name__ == "__main__":
    main()
