
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

        # Prepare gallery items as tuples (image_path, caption)
        gallery_items = []

        for result in results:
            image_path = result['image_path']
            # Fix path - remove ../ if present
            if image_path.startswith('../'):
                image_path = image_path[3:]  # Remove ../

            if os.path.exists(image_path):
                caption = f"🎯 Similarity: {result['similarity']:.3f}
📝 {result['caption']}"
                gallery_items.append((image_path, caption))
            else:
                # For missing images, we can't add them to the gallery
                pass

        return gallery_items, f"Found {len(gallery_items)} results for: '{query}'"

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

        # Prepare gallery items as tuples (image_path, caption)
        gallery_items = []

        for result in results:
            image_path = result['image_path']
            # Fix path - remove ../ if present
            if image_path.startswith('../'):
                image_path = image_path[3:]  # Remove ../

            if os.path.exists(image_path):
                caption = f"🎯 Similarity: {result['similarity']:.3f}
📝 {result['caption']}"
                gallery_items.append((image_path, caption))
            else:
                # For missing images, we can't add them to the gallery
                pass

        return gallery_items, f"Found {len(gallery_items)} similar descriptions"

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

    with gr.Blocks(
        title="🔍 Multimodal Search Engine", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
            min-height: 100vh !important;
        }

        /* Modern header styling */
        .gradio-container h1 {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center !important;
            margin: 2rem 0 !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

        /* Card styling */
        .card {
            background: white !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(10px) !important;
            margin: 1rem 0 !important;
        }

        /* Button styling */
        .btn {
            border-radius: 16px !important;
            font-weight: 700 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 0.875rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        .btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important;
        }

        .btn-primary {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
        }

        /* Input styling */
        .input {
            border-radius: 16px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 1rem 1.25rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px) !important;
            font-size: 1rem !important;
        }

        .input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
            background: white !important;
        }

        /* Gallery styling */
        .gallery {
            border-radius: 20px !important;
            overflow: hidden !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important;
            background: white !important;
            padding: 1rem !important;
        }

        .gallery img {
            border-radius: 16px !important;
            transition: transform 0.3s ease !important;
        }

        .gallery img:hover {
            transform: scale(1.02) !important;
        }

        /* Tab styling */
        .tab-nav {
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 20px !important;
            padding: 0.5rem !important;
            margin: 2rem 0 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }

        .tab-nav button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            padding: 0.75rem 1.5rem !important;
            margin: 0.25rem !important;
        }

        .tab-nav button.selected {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }

        /* Status messages */
        .status {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            padding: 1rem 1.5rem !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }

        /* Dataset info cards */
        .dataset-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin: 0.5rem 0 !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 1rem !important;
            }

            .gradio-container h1 {
                font-size: 2.5rem !important;
            }

            .card {
                padding: 1.5rem !important;
                margin: 0.5rem 0 !important;
            }
        }
        """
    ) as app:
        gr.Markdown(description)

        # Dataset information
        with gr.Row():
            with gr.Column(scale=1):
                # Get values and format properly
                num_images = model_info.get('num_images', 'Unknown')
                num_embeddings = model_info.get('total_embeddings', model_info.get('num_samples', 'Unknown'))
                embedding_dim = model_info.get('embedding_dim', 'Unknown')
                model_name = model_info.get('model_name', 'Unknown')
                dataset = model_info.get('dataset', 'Unknown')
                processing_date = model_info.get('processing_date', 'Unknown')

                # Format numbers properly
                images_text = f"{num_images:,}" if isinstance(num_images, int) else str(num_images)
                embeddings_text = f"{num_embeddings:,}" if isinstance(num_embeddings, int) else str(num_embeddings)
                model_display = model_name.split('/')[-1] if '/' in model_name else model_name

                gr.Markdown(f"""
                <div class="dataset-card">
                <h3 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.25rem; font-weight: 700;">📊 Dataset Information</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.875rem;">
                    <div><strong>📸 Images:</strong> {images_text}</div>
                    <div><strong>🔢 Embeddings:</strong> {embeddings_text}</div>
                    <div><strong>📐 Dimension:</strong> {embedding_dim}D</div>
                    <div><strong>🤖 Model:</strong> {model_display}</div>
                    <div><strong>📁 Dataset:</strong> {dataset}</div>
                    <div><strong>📅 Date:</strong> {processing_date}</div>
                </div>
                </div>
                """)

            with gr.Column(scale=1):
                gr.Markdown("""
                <div class="dataset-card">
                <h3 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.25rem; font-weight: 700;">🔥 Popular Searches</h3>
                <p style="margin: 0 0 1rem 0; color: #6b7280; font-size: 0.875rem;">Try these popular search terms:</p>
                """)

                # Create a simple list of popular searches
                popular_text = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">'
                for search in popular_searches:
                    popular_text += f'<span style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 0.5rem 1rem; border-radius: 12px; font-size: 0.875rem; font-weight: 600; display: inline-block; margin: 0.25rem;">{search}</span>'
                popular_text += '</div></div>'

                gr.Markdown(popular_text)

        # Main search interface
        with gr.Tabs():
            # Text-to-Image Search Tab
            with gr.Tab("🔤 Text-to-Image Search"):
                gr.Markdown("""
                <div class="card">
                <h2 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.5rem; font-weight: 700;">🔤 Text-to-Image Search</h2>
                <p style="margin: 0 0 2rem 0; color: #6b7280; font-size: 1rem;">Describe what you're looking for and discover relevant images from the dataset</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        text_query = gr.Textbox(
                            label="🔍 Search Query",
                            placeholder="e.g., 'a dog playing in the park' or 'children smiling'",
                            info="💡 Be specific! Try describing objects, actions, colors, or emotions. The more descriptive, the better the results!",
                            elem_classes=["input"]
                        )
                        num_results_text = gr.Slider(
                            label="📊 Number of Results",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            info="Choose how many results to display"
                        )
                        search_btn = gr.Button("🔍 Search Images", variant="primary", elem_classes=["btn", "btn-primary"])

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        <div class="dataset-card">
                        <h3 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.125rem; font-weight: 700;">🔥 Popular Searches</h3>
                        <p style="margin: 0 0 1rem 0; color: #6b7280; font-size: 0.875rem;">Click any suggestion to search:</p>
                        """)

                        # Create clickable search suggestion buttons
                        with gr.Row():
                            with gr.Column():
                                for i, search in enumerate(popular_searches[:4]):  # First 4 searches
                                    btn = gr.Button(
                                        f"🔍 {search}", 
                                        size="sm", 
                                        variant="secondary",
                                        elem_classes=["btn"]
                                    )
                                    btn.click(
                                        lambda s=search: s, 
                                        outputs=text_query
                                    )

                        with gr.Row():
                            with gr.Column():
                                for i, search in enumerate(popular_searches[4:]):  # Last 4 searches
                                    btn = gr.Button(
                                        f"🔍 {search}", 
                                        size="sm", 
                                        variant="secondary",
                                        elem_classes=["btn"]
                                    )
                                    btn.click(
                                        lambda s=search: s, 
                                        outputs=text_query
                                    )

                        gr.Markdown("""
                        <div class="dataset-card" style="margin-top: 1rem;">
                        <h3 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.125rem; font-weight: 700;">💡 Search Tips</h3>
                        <div style="color: #6b7280; font-size: 0.875rem; line-height: 1.6;">
                        <strong>Try searching for:</strong><br>
                        • <strong>Animals:</strong> 'dog', 'cat', 'bird', 'horse'<br>
                        • <strong>Activities:</strong> 'playing', 'running', 'cooking'<br>
                        • <strong>Objects:</strong> 'car', 'house', 'food'<br>
                        • <strong>Emotions:</strong> 'smiling', 'happy', 'sad'<br>
                        • <strong>Scenes:</strong> 'beach', 'park', 'kitchen'
                        </div>
                        </div>
                        """)

                # Results
                text_results = gr.Gallery(
                    label="🎨 Search Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    elem_classes=["gallery"]
                )
                text_status = gr.Textbox(
                    label="📊 Status", 
                    interactive=False,
                    elem_classes=["status"]
                )

                # Connect search button
                search_btn.click(
                    search_images,
                    inputs=[text_query, num_results_text],
                    outputs=[text_results, text_status]
                )

            # Image-to-Text Search Tab
            with gr.Tab("🖼️ Image-to-Text Search"):
                gr.Markdown("""
                <div class="card">
                <h2 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.5rem; font-weight: 700;">🖼️ Image-to-Text Search</h2>
                <p style="margin: 0 0 2rem 0; color: #6b7280; font-size: 1rem;">Upload an image to find similar text descriptions from the dataset</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(
                            label="📁 Upload Image",
                            type="pil",
                            info="💡 Upload a clear image with a main subject for best search results!",
                            elem_classes=["input"]
                        )
                        num_results_image = gr.Slider(
                            label="📊 Number of Results",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            info="Choose how many results to display"
                        )
                        search_img_btn = gr.Button("🔍 Search Descriptions", variant="primary", elem_classes=["btn", "btn-primary"])

                    with gr.Column(scale=1):
                        gr.Markdown("""
                        <div class="dataset-card">
                        <h3 style="margin: 0 0 1rem 0; color: #374151; font-size: 1.125rem; font-weight: 700;">📋 Upload Guidelines</h3>
                        <div style="color: #6b7280; font-size: 0.875rem; line-height: 1.6;">
                        <strong>Supported formats:</strong><br>
                        • JPG, JPEG<br>
                        • PNG<br>
                        • BMP, GIF<br><br>

                        <strong>Best results with:</strong><br>
                        • Clear, well-lit images<br>
                        • Single main subject<br>
                        • Good contrast
                        </div>
                        </div>
                        """)

                # Results
                image_results = gr.Gallery(
                    label="🎨 Search Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    elem_classes=["gallery"]
                )
                image_status = gr.Textbox(
                    label="📊 Status", 
                    interactive=False,
                    elem_classes=["status"]
                )

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

