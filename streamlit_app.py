
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
    image_embeddings = np.load('../embeddings/image_embeddings.npy')
    text_embeddings = np.load('../embeddings/text_embeddings.npy')

    # Load metadata
    metadata = pd.read_csv('../embeddings/metadata.csv')

    # Load model info
    with open('../embeddings/model_info.json', 'r') as f:
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
        page_title="🔍 Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("🔍 Search Engine")
    st.markdown("A powerful multimodal search engine using OpenAI CLIP")

    # Sidebar
    st.sidebar.header("Search Options")

    # Search type selection
    search_type = st.sidebar.radio(
        "Choose search type:",
        ["Text-to-Image Search", "Image-to-Text Search"]
    )

    # Number of results
    top_k = st.sidebar.slider(
        "Number of results:",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top results to display"
    )

    # Popular searches
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔥 Popular Searches")

    popular_searches = [
        "dog playing", "children smiling", "red car", "food cooking",
        "person running", "cat sleeping", "blue sky", "water beach",
        "house building", "tree nature", "person walking", "animal pet"
    ]

    # Create clickable search suggestions
    for i, search in enumerate(popular_searches):
        if st.sidebar.button(f"🔍 {search}", key=f"popular_{i}"):
            st.session_state.popular_search = search
            st.session_state.auto_search = True

    # Display dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Information")

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

    st.sidebar.metric("Total Images", images_text)
    st.sidebar.metric("Total Embeddings", embeddings_text)
    st.sidebar.metric("Embedding Dimension", f"{embedding_dim}D")
    st.sidebar.metric("Model", model_display)
    st.sidebar.metric("Dataset", dataset)
    st.sidebar.metric("Processing Date", processing_date)

    # Check if this is a demo dataset
    num_images = model_info.get('num_images', len(metadata))
    if isinstance(num_images, int) and num_images < 1000:
        st.warning(f"⚠️ **Demo Mode**: You're using a small subset ({num_images:,} images) of the full Flickr8k dataset. For production use, run the full dataset processing in Part 1 to get all 8,091 images.")

    # Main content area
    if search_type == "Text-to-Image Search":
        st.header("🔤 Text-to-Image Search")
        st.markdown("Enter a text description to find similar images:")

        # Search suggestions
        st.markdown("#### 💡 Search Tips")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Try searching for:**")
            st.markdown("• Animals: 'dog', 'cat', 'bird'")
            st.markdown("• Activities: 'playing', 'running', 'cooking'")
            st.markdown("• Objects: 'car', 'house', 'food'")
            st.markdown("• Emotions: 'smiling', 'happy', 'sad'")

        with col2:
            st.markdown("**Example queries:**")
            if st.button("🐕 A dog playing", key="example1"):
                st.session_state.example_query = "a dog playing"
                st.session_state.auto_search = True
            if st.button("👶 Children smiling", key="example2"):
                st.session_state.example_query = "children smiling"
                st.session_state.auto_search = True
            if st.button("🚗 Red car", key="example3"):
                st.session_state.example_query = "red car"
                st.session_state.auto_search = True
            if st.button("🍕 Food cooking", key="example4"):
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

    else:  # Image-to-Text Search
        st.header("🖼️ Image-to-Text Search")
        st.markdown("Upload an image to find similar text descriptions:")

        # Upload guidance
        st.markdown("#### 📋 Upload Guidelines")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Supported formats:**")
            st.markdown("• JPG, JPEG")
            st.markdown("• PNG")
            st.markdown("• BMP, GIF")

        with col2:
            st.markdown("**Best results with:**")
            st.markdown("• Clear, well-lit images")
            st.markdown("• Single main subject")
            st.markdown("• Good contrast")

        # Image upload
        uploaded_file = st.file_uploader(
            "📁 Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="💡 Upload a clear image with a main subject for best search results!",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Search Descriptions", type="primary"):
                with st.spinner("Searching for similar descriptions..."):
                    results = image_to_text_search(uploaded_image, top_k)

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

if __name__ == "__main__":
    main()
