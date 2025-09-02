import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Multimodal Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.search-section {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.result-card {
    background-color: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.similarity-score {
    font-weight: bold;
    color: #2e8b57;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load the CLIP model and embeddings (cached for performance)"""
    # Load model
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Load embeddings and metadata
    image_embeddings = np.load('embeddings/image_embeddings.npy')
    text_embeddings = np.load('embeddings/text_embeddings.npy')
    metadata = pd.read_csv('embeddings/metadata.csv')
    
    return model, processor, image_embeddings, text_embeddings, metadata

class StreamlitSearchEngine:
    def __init__(self, model, processor, image_embeddings, text_embeddings, metadata):
        self.model = model
        self.processor = processor
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.metadata = metadata
    
    def embed_text(self, text):
        """Generate embedding for text input"""
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten()
        except Exception as e:
            st.error(f"Error processing text: {e}")
            return None
    
    def embed_image(self, image):
        """Generate embedding for image input"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def text_to_image_search(self, query, top_k=5):
        """Search for images using text query"""
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return None
        
        similarities = cosine_similarity([query_embedding], self.image_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'image_id': self.metadata.iloc[idx]['image_id'],
                'image_path': self.metadata.iloc[idx]['image_path'],
                'caption': self.metadata.iloc[idx]['caption'],
                'similarity_score': similarities[idx]
            }
            results.append(result)
        
        return results
    
    def image_to_text_search(self, image, top_k=5):
        """Search for text descriptions using image query"""
        query_embedding = self.embed_image(image)
        if query_embedding is None:
            return None
        
        similarities = cosine_similarity([query_embedding], self.text_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'image_id': self.metadata.iloc[idx]['image_id'],
                'image_path': self.metadata.iloc[idx]['image_path'],
                'caption': self.metadata.iloc[idx]['caption'],
                'similarity_score': similarities[idx]
            }
            results.append(result)
        
        return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced Multimodal Search Engine</h1>', unsafe_allow_html=True)
    
    # Sidebar with project information
    with st.sidebar:
        st.markdown("## 📋 Project Overview")
        st.markdown("""
        This is an advanced multimodal search engine that supports **bidirectional search**:
        
        - **🔍 Text-to-Image**: Find images based on text descriptions
        - **🖼️ Image-to-Text**: Find text descriptions based on uploaded images
        
        ### 🛠️ Technology Stack:
        - **CLIP Model**: OpenAI's multimodal model for text-image understanding
        - **Streamlit**: Web application framework
        - **PyTorch**: Deep learning framework
        - **Transformers**: Hugging Face model library
        - **Scikit-learn**: Similarity calculations
        - **PIL**: Image processing
        
        ### 🎯 Key Features:
        - Real-time multimodal search
        - Confidence scoring
        - Visual result display
        - Interactive interface
        - Performance analytics
        """)
        
        st.markdown("## 📊 Dataset Info")
        st.markdown("""
        - **Images**: 10 sample images
        - **Captions**: 10 corresponding descriptions
        - **Embeddings**: 512-dimensional vectors
        - **Model**: CLIP ViT-Base-Patch32
        """)
    
    # Load model and data
    with st.spinner('Loading model and data...'):
        model, processor, image_embeddings, text_embeddings, metadata = load_model_and_data()
        search_engine = StreamlitSearchEngine(model, processor, image_embeddings, text_embeddings, metadata)
    
    st.success('✅ Model and data loaded successfully!')
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Text-to-Image Search", "🖼️ Image-to-Text Search", "📊 Analytics"])
    
    with tab1:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Text-to-Image Search")
        st.markdown("Enter a text description to find the most similar images.")
        
        # Text input
        text_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'dog running', 'beautiful sunset', 'person cooking'...",
            help="Describe what you're looking for in the images"
        )
        
        # Search button
        if st.button("🔍 Search Images", type="primary"):
            if text_query:
                with st.spinner('Searching for images...'):
                    results = search_engine.text_to_image_search(text_query, top_k=5)
                
                if results:
                    st.markdown(f"### 📊 Results for: '{text_query}'")
                    
                    # Display results in columns
                    cols = st.columns(2)
                    for i, result in enumerate(results):
                        with cols[i % 2]:
                            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                            
                            # Display image
                            try:
                                if os.path.exists(result['image_path']):
                                    image = Image.open(result['image_path'])
                                    st.image(image, caption=f"#{result['rank']} {result['image_id']}", use_container_width=True)
                                else:
                                    st.error(f"Image file not found: {result['image_path']}")
                            except Exception as e:
                                st.error(f"Could not load image {result['image_id']}: {str(e)}")
                            
                            # Display caption and similarity
                            st.markdown(f"**Caption:** {result['caption']}")
                            st.markdown(f"**Similarity:** <span class='similarity-score'>{result['similarity_score']:.3f}</span>", unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No results found. Please try a different query.")
            else:
                st.warning("Please enter a search query.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Image-to-Text Search")
        st.markdown("Upload an image to find the most similar text descriptions.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to find similar text descriptions"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Search button
            if st.button("🔍 Search Descriptions", type="primary"):
                with st.spinner('Searching for similar descriptions...'):
                    results = search_engine.image_to_text_search(image, top_k=5)
                
                if results:
                    st.markdown("### 📊 Most Similar Descriptions")
                    
                    # Display results
                    for result in results:
                        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                        
                        # Display image and caption
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            try:
                                if os.path.exists(result['image_path']):
                                    result_image = Image.open(result['image_path'])
                                    st.image(result_image, caption=result['image_id'], use_container_width=True)
                                else:
                                    st.error(f"Image file not found: {result['image_path']}")
                            except Exception as e:
                                st.error(f"Could not load image {result['image_id']}: {str(e)}")
                        
                        with col2:
                            st.markdown(f"**Rank:** #{result['rank']}")
                            st.markdown(f"**Description:** {result['caption']}")
                            st.markdown(f"**Similarity:** <span class='similarity-score'>{result['similarity_score']:.3f}</span>", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No results found. Please try a different image.")
        else:
            st.info("👆 Please upload an image to search for similar descriptions.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📊 Search Analytics")
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", len(metadata))
        with col2:
            st.metric("Embedding Dimension", "512")
        with col3:
            st.metric("Model", "CLIP ViT-Base")
        with col4:
            st.metric("Search Types", "2 (Text↔Image)")
        
        # Sample data display
        st.markdown("### 📋 Dataset Sample")
        st.dataframe(metadata.head(10), use_container_width=True)
        
        # Model information
        st.markdown("### 🤖 Model Information")
        st.markdown("""
        **CLIP (Contrastive Language-Image Pre-training)**
        
        - **Architecture**: Vision Transformer (ViT) + Text Encoder
        - **Training**: Contrastive learning on 400M image-text pairs
        - **Capabilities**: Understanding relationships between images and text
        - **Embedding Space**: 512-dimensional shared representation
        - **Use Cases**: Image search, text search, multimodal understanding
        """)

if __name__ == "__main__":
    main()
