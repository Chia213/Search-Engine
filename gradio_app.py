
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

# Text-to-Image Search Interface
def search_images(query, num_results):
    """Gradio interface for text-to-image search"""
    if not query.strip():
        return [], "Please enter a search query."

    try:
        results = text_to_image_search(query, num_results)

        if not results:
            return [], "No results found. Try a different search query."

        # Prepare images and captions for display
        images = []
        captions = []

        for result in results:
            image_path = result['image_path']
            # Fix path - remove ../ if present
            if image_path.startswith('../'):
                image_path = image_path[3:]  # Remove ../

            if os.path.exists(image_path):
                images.append(image_path)
                captions.append(f"Similarity: {result['similarity']:.3f}\nCaption: {result['caption']}")
            else:
                images.append(None)
                captions.append(f"Image not found: {image_path}")

        return images, f"Found {len(results)} results for: '{query}'"

    except Exception as e:
        return [], f"Error during search: {str(e)}"

# Image-to-Text Search Interface
def search_descriptions(image, num_results):
    """Gradio interface for image-to-text search"""
    if image is None:
        return [], "Please upload an image."

    try:
        results = image_to_text_search(image, num_results)

        if not results:
            return [], "No results found. Try a different image."

        # Prepare images and captions for display
        images = []
        captions = []

        for result in results:
            image_path = result['image_path']
            # Fix path - remove ../ if present
            if image_path.startswith('../'):
                image_path = image_path[3:]  # Remove ../

            if os.path.exists(image_path):
                images.append(image_path)
                captions.append(f"Similarity: {result['similarity']:.3f}\nCaption: {result['caption']}")
            else:
                images.append(None)
                captions.append(f"Image not found: {image_path}")

        return images, f"Found {len(results)} similar descriptions"

    except Exception as e:
        return [], f"Error during search: {str(e)}"

# Create Gradio interface
def create_gradio_app():
    """Create the Gradio web application"""

    # Project description
    description = """
    # 🔍 Multimodal Search Engine

    A powerful search engine that can find images using text descriptions and find text descriptions using images.

    **Technology Stack:**
    - **Model**: OpenAI CLIP (Contrastive Language-Image Pre-training)
    - **Framework**: Gradio for web interface
    - **Dataset**: Flickr8k (8,091 images with captions)
    - **Embeddings**: 512-dimensional vector representations
    - **Similarity**: Cosine similarity for matching

    **Features:**
    - Text-to-Image Search: Describe what you're looking for
    - Image-to-Text Search: Upload an image to find similar descriptions
    - Real-time similarity scoring
    - Interactive web interface
    """

    # Popular search suggestions
    popular_searches = [
        "dog playing", "children smiling", "red car", "food cooking",
        "person running", "cat sleeping", "blue sky", "water beach"
    ]

    with gr.Blocks(title="🔍 Search Engine", theme=gr.themes.Soft()) as app:
        gr.Markdown(description)

        # Dataset information
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"""
                ### 📊 Dataset Information
                - **Total Images**: {model_info.get('num_images', 'Unknown'):,}
                - **Total Embeddings**: {model_info.get('total_embeddings', model_info.get('num_samples', 'Unknown')):,}
                - **Embedding Dimension**: {model_info.get('embedding_dim', 'Unknown')}D
                - **Model**: {model_info.get('model_name', 'Unknown').split('/')[-1]}
                - **Dataset**: {model_info.get('dataset', 'Unknown')}
                - **Processing Date**: {model_info.get('processing_date', 'Unknown')}
                """)

            with gr.Column(scale=1):
                gr.Markdown(f"""
                ### 🔥 Popular Searches
                Click any suggestion to search:
                """)
                # Create clickable search suggestions
                for i, search in enumerate(popular_searches):
                    if i % 2 == 0:
                        with gr.Row():
                            gr.Button(f"🔍 {search}", size="sm").click(
                                lambda s=search: s, outputs=gr.Textbox(visible=False)
                            ).then(
                                search_images, 
                                inputs=[gr.Textbox(value=search, visible=False), gr.Slider(1, 20, 5)],
                                outputs=[gr.Gallery(), gr.Textbox()]
                            )
                    else:
                        gr.Button(f"🔍 {search}", size="sm").click(
                            lambda s=search: s, outputs=gr.Textbox(visible=False)
                        ).then(
                            search_images,
                            inputs=[gr.Textbox(value=search, visible=False), gr.Slider(1, 20, 5)],
                            outputs=[gr.Gallery(), gr.Textbox()]
                        )

        # Main search interface
        with gr.Tabs():
            # Text-to-Image Search Tab
            with gr.Tab("🔤 Text-to-Image Search"):
                gr.Markdown("Enter a text description to find similar images:")

                with gr.Row():
                    with gr.Column(scale=3):
                        text_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., 'a dog playing in the park' or 'children smiling'",
                            info="Describe what you're looking for in the images"
                        )
                        num_results_text = gr.Slider(
                            label="Number of Results",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                        search_btn = gr.Button("🔍 Search Images", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Search Tips
                        **Try searching for:**
                        - Animals: 'dog', 'cat', 'bird'
                        - Activities: 'playing', 'running', 'cooking'
                        - Objects: 'car', 'house', 'food'
                        - Emotions: 'smiling', 'happy', 'sad'
                        """)

                # Results
                text_results = gr.Gallery(
                    label="Search Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
                text_status = gr.Textbox(label="Status", interactive=False)

                # Connect search button
                search_btn.click(
                    search_images,
                    inputs=[text_query, num_results_text],
                    outputs=[text_results, text_status]
                )

            # Image-to-Text Search Tab
            with gr.Tab("🖼️ Image-to-Text Search"):
                gr.Markdown("Upload an image to find similar text descriptions:")

                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            info="Upload a clear image with a main subject for best results"
                        )
                        num_results_image = gr.Slider(
                            label="Number of Results",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                        search_img_btn = gr.Button("🔍 Search Descriptions", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 📋 Upload Guidelines
                        **Supported formats:**
                        - JPG, JPEG
                        - PNG
                        - BMP, GIF

                        **Best results with:**
                        - Clear, well-lit images
                        - Single main subject
                        - Good contrast
                        """)

                # Results
                image_results = gr.Gallery(
                    label="Search Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
                image_status = gr.Textbox(label="Status", interactive=False)

                # Connect search button
                search_img_btn.click(
                    search_descriptions,
                    inputs=[image_input, num_results_image],
                    outputs=[image_results, image_status]
                )

        # Footer
        gr.Markdown("""
        ---
        **🔍 Search Engine** - Built with Gradio and OpenAI CLIP
        """)

    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
