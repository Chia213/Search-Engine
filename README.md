# 🔍 Search Engine

A sophisticated AI-powered search engine that finds images based on text descriptions and text descriptions based on uploaded images. Built using CLIP model and advanced machine learning techniques.

## ✨ Features

- **🔍 Bidirectional Search**: Text-to-image and image-to-text capabilities
- **🤖 Advanced AI**: CLIP model with intent analysis and query expansion
- **🌐 Web Interface**: Professional Streamlit and Gradio applications
- **📊 Analytics**: Performance metrics and visualizations
- **⚡ Real-time**: Instant search results with confidence scoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd Search-Engine

# Install dependencies
pip install -r requirements.txt
```

### 📥 Dataset Setup

**Important**: Download the Flickr8k dataset separately (not included due to size):

1. **Images**: [Flickr8k_Dataset.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
2. **Text**: [Flickr8k_text.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

```bash
# Setup data structure
mkdir -p data/images
unzip Flickr8k_Dataset.zip -d data/images/
unzip Flickr8k_text.zip
mv Flickr8k.token.txt data/
cp data/Flickr8k.token.txt data/captions.txt
```

### 🏃‍♂️ Running the Application

```bash
# Generate embeddings (run once)
jupyter notebook notebook/Part1_Data_Preparation_Embedding.ipynb

# Launch web app
streamlit run streamlit_app.py
# OR
python gradio_app.py
```

Visit `http://localhost:8501` (Streamlit) or `http://localhost:7860` (Gradio)

## 📁 Project Structure

```
Search-Engine/
├── notebook/                     # Jupyter notebooks
│   ├── Part1_Data_Preparation_Embedding.ipynb
│   ├── Part2_Search_Functionality.ipynb
│   └── Part3_Multimodal_Interface.ipynb
├── data/                        # Dataset (download separately)
│   ├── images/                  # 8,091 images
│   └── captions.txt
├── embeddings/                  # Generated embeddings
├── streamlit_app.py            # Streamlit web app
├── gradio_app.py               # Gradio web app
└── requirements.txt            # Dependencies
```

## 🛠️ Technology Stack

- **AI Model**: CLIP ViT-Base-Patch32 (via Hugging Face Transformers)
- **Framework**: PyTorch, Transformers
- **Web Apps**: Streamlit, Gradio
- **Data Processing**: NumPy, Pandas, PIL
- **ML**: Scikit-learn, Cosine Similarity

## 🔧 Troubleshooting

**Missing embeddings**: Run Part 1 notebook first
**Module errors**: `pip install -r requirements.txt`
**Memory issues**: Ensure 8GB+ RAM available
**Path errors**: Verify data structure matches above

## 📄 License

Educational project for academic course in Machine Learning and Deep Learning.
