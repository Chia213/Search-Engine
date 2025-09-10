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
        return [], "Please enter a search query."
    
    results = text_to_image_search(query, int(top_k))
    
    if not results:
        return [], "No results found."
    
    # Prepare images and captions for display
    images = []
    captions = []
    detailed_results = []
    
    for result in results:
        image_path = result['image_path']
        if os.path.exists(image_path):
            images.append(image_path)
            captions.append(f"Rank {result['rank']} (Similarity: {result['similarity']:.3f})\n{result['caption']}")
            detailed_results.append(f"**Rank {result['rank']}** (Similarity: {result['similarity']:.3f})\n📝 **Caption:** {result['caption']}\n🖼️ **Image:** {image_path}\n")
        else:
            # If image doesn't exist, create a placeholder
            images.append(None)
            captions.append(f"Rank {result['rank']} (Similarity: {result['similarity']:.3f})\n{result['caption']}\n❌ Image not found: {image_path}")
            detailed_results.append(f"**Rank {result['rank']}** (Similarity: {result['similarity']:.3f})\n📝 **Caption:** {result['caption']}\n❌ **Image not found:** {image_path}\n")
    
    # Combine all detailed results
    detailed_text = f"🔍 **Search Results for: '{query}'** - Found {len(images)} results\n\n" + "\n".join(detailed_results)
    
    return images, detailed_text

def search_image_to_text(image, top_k):
    """Gradio interface for image-to-text search"""
    if image is None:
        return [], "Please upload an image."
    
    results = image_to_text_search(image, int(top_k))
    
    if not results:
        return [], "No results found."
    
    # Prepare images and captions for display
    images = []
    captions = []
    detailed_results = []
    
    for result in results:
        image_path = result['image_path']
        if os.path.exists(image_path):
            images.append(image_path)
            captions.append(f"Rank {result['rank']} (Similarity: {result['similarity']:.3f})\n{result['caption']}")
            detailed_results.append(f"**Rank {result['rank']}** (Similarity: {result['similarity']:.3f})\n📝 **Caption:** {result['caption']}\n🖼️ **Image:** {image_path}\n")
        else:
            # If image doesn't exist, create a placeholder
            images.append(None)
            captions.append(f"Rank {result['rank']} (Similarity: {result['similarity']:.3f})\n{result['caption']}\n❌ Image not found: {image_path}")
            detailed_results.append(f"**Rank {result['rank']}** (Similarity: {result['similarity']:.3f})\n📝 **Caption:** {result['caption']}\n❌ **Image not found:** {image_path}\n")
    
    # Combine all detailed results
    detailed_text = f"🔍 **Search Results for Uploaded Image** - Found {len(images)} results\n\n" + "\n".join(detailed_results)
    
    return images, detailed_text

# Create Gradio Interface
def create_gradio_app():
    """Create the Gradio application"""
    
    with gr.Blocks(title="🔍 Multimodal Search Engine") as app:
        gr.Markdown("# 🔍 Multimodal Search Engine")
        gr.Markdown("Search for images using text descriptions or find text descriptions using uploaded images.")
        
        with gr.Tabs():
            # Text-to-Image Search Tab
            with gr.Tab("🔤 Text to Image Search"):
                gr.Markdown("Enter a text description to find similar images.")
                
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter a description (e.g., 'a dog running in the park')",
                            lines=2
                        )
                        top_k_text = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Results"
                        )
                        text_search_btn = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column():
                        text_output = gr.Markdown(label="Search Results")
                        text_gallery = gr.Gallery(
                            label="Search Results",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                
                text_search_btn.click(
                    fn=search_text_to_image,
                    inputs=[text_input, top_k_text],
                    outputs=[text_gallery, text_output]
                )
            
            # Image-to-Text Search Tab
            with gr.Tab("🖼️ Image to Text Search"):
                gr.Markdown("Upload an image to find similar text descriptions.")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image",
                            type="filepath"
                        )
                        top_k_image = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Results"
                        )
                        image_search_btn = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Markdown(label="Search Results")
                        image_gallery = gr.Gallery(
                            label="Search Results",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                
                image_search_btn.click(
                    fn=search_image_to_text,
                    inputs=[image_input, top_k_image],
                    outputs=[image_gallery, image_output]
                )
        
        # Dataset Information
        with gr.Row():
            gr.Markdown(f"""
            ### 📊 Dataset Information
            - **Total Images**: {model_info.get('num_images', 'Unknown'):,}
            - **Total Embeddings**: {model_info.get('total_embeddings', model_info.get('num_samples', 'Unknown')):,}
            - **Embedding Dimension**: {model_info.get('embedding_dim', 'Unknown')}D
            - **Model**: {model_info.get('model_name', 'Unknown').split('/')[-1]}
            - **Dataset**: {model_info.get('dataset', 'Unknown')}
            - **Processing Date**: {model_info.get('processing_date', 'Unknown')}
            """)
    
    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)