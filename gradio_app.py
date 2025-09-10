import gradio as gr
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CLIP model
def load_clip_model():
    """Load CLIP model and processor"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Load embeddings data
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

    # Prepare results
    results = []
    for i, idx in enumerate(top_indices):
        similarity_score = similarities[idx]
        image_path = metadata.iloc[idx]['image_path']
        caption = metadata.iloc[idx]['caption']
        
        results.append({
            'image_path': image_path,
            'caption': caption,
            'similarity': similarity_score,
            'rank': i + 1
        })

    return results

# Image-to-Text Search Function
def image_to_text_search(uploaded_image, top_k=5):
    """Search for text descriptions based on uploaded image"""
    if uploaded_image is None:
        return []

    # Preprocess image
    image = Image.open(uploaded_image).convert('RGB')
    inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    # Calculate similarities with all text embeddings
    similarities = cosine_similarity(image_embedding.cpu().numpy(), text_embeddings)[0]

    # Get top-k most similar texts
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Prepare results
    results = []
    for i, idx in enumerate(top_indices):
        similarity_score = similarities[idx]
        image_path = metadata.iloc[idx]['image_path']
        caption = metadata.iloc[idx]['caption']
        
        results.append({
            'image_path': image_path,
            'caption': caption,
            'similarity': similarity_score,
            'rank': i + 1
        })

    return results

# Gradio Interface Functions
def search_text_to_image(query, top_k):
    """Gradio interface for text-to-image search"""
    if not query.strip():
        return "Please enter a search query."
    
    results = text_to_image_search(query, int(top_k))
    
    if not results:
        return "No results found."
    
    # Format results
    output = f"🔍 **Search Results for: '{query}'**\n\n"
    for result in results:
        output += f"**Rank {result['rank']}** (Similarity: {result['similarity']:.3f})\n"
        output += f"📝 Caption: {result['caption']}\n"
        output += f"🖼️ Image: {result['image_path']}\n\n"
    
    return output

def search_image_to_text(image, top_k):
    """Gradio interface for image-to-text search"""
    if image is None:
        return "Please upload an image."
    
    results = image_to_text_search(image, int(top_k))
    
    if not results:
        return "No results found."
    
    # Format results
    output = f"🔍 **Search Results for Uploaded Image**\n\n"
    for result in results:
        output += f"**Rank {result['rank']}** (Similarity: {result['similarity']:.3f})\n"
        output += f"📝 Caption: {result['caption']}\n"
        output += f"🖼️ Image: {result['image_path']}\n\n"
    
    return output

# Create Gradio Interface using simple Interface
def create_gradio_app():
    """Create the Gradio application using simple Interface"""
    
    # Text-to-Image Interface
    text_interface = gr.Interface(
        fn=search_text_to_image,
        inputs=[
            gr.Textbox(label="Search Query", placeholder="Enter a description (e.g., 'a dog running in the park')"),
            gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results")
        ],
        outputs=gr.Textbox(label="Search Results", lines=10),
        title="🔤 Text to Image Search",
        description="Enter a text description to find similar images."
    )
    
    # Image-to-Text Interface
    image_interface = gr.Interface(
        fn=search_image_to_text,
        inputs=[
            gr.Image(label="Upload Image", type="filepath"),
            gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results")
        ],
        outputs=gr.Textbox(label="Search Results", lines=10),
        title="🖼️ Image to Text Search",
        description="Upload an image to find similar text descriptions."
    )
    
    # Combine interfaces
    app = gr.TabbedInterface(
        [text_interface, image_interface],
        ["Text to Image", "Image to Text"],
        title="🔍 Multimodal Search Engine"
    )
    
    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)