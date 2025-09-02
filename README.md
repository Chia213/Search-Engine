# Search Engine Project

This project implements a multimodal search engine that can find images based on text queries and text based on image queries using CLIP embeddings.

## Project Structure

```
Search-Engine/
├── Part1_Data_Preparation_Embedding.ipynb  # Part 1: Data preparation and embedding generation
├── requirements.txt                         # Python dependencies
├── README.md                               # This file
├── data/                                   # Dataset directory
│   ├── images/                            # Image files
│   └── captions.txt                       # Image captions
└── embeddings/                            # Generated embeddings
    ├── image_embeddings.npy               # Image embeddings
    ├── text_embeddings.npy                # Text embeddings
    ├── metadata.csv                       # Metadata for embeddings
    └── model_info.json                    # Model information
```

## Part 1: Data Preparation & Embedding

This notebook implements:
1. **Dataset Loading**: Load and explore the dataset (Flickr8k or custom dataset)
2. **Model Selection**: Use CLIP (Contrastive Language-Image Pre-training) from OpenAI
3. **Embedding Generation**: Generate vector embeddings for both images and text
4. **Data Storage**: Store embeddings with corresponding metadata

### Key Features:
- Uses CLIP model for multimodal embeddings
- Normalizes embeddings for better similarity calculations
- Creates sample dataset for demonstration
- Saves all embeddings and metadata for future use

### Requirements:
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- PIL/Pillow
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter notebook:
   ```bash
   jupyter notebook Part1_Data_Preparation_Embedding.ipynb
   ```

2. Execute all cells in order to generate embeddings

## Next Steps

- Part 2: Search Functionality (text-to-image search)
- Part 3: Multimodal Interface (image-to-text search + web app)

## Model Information

- **Model**: CLIP ViT-Base-Patch32
- **Provider**: OpenAI via Hugging Face
- **Embedding Dimension**: 512
- **Capabilities**: Multimodal (text + image) understanding
