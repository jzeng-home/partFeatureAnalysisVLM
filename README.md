---

# Part Feature Analysis / Labeling via VLM

This project implements a comprehensive pipeline for analyzing and labeling 3D part features using Vision-Language Models (VLMs). The system transforms 3D objects into detailed textual descriptions through a multi-stage process involving 3D rendering, AI-powered caption generation, and intelligent caption selection.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Technical Deep Dive](#technical-deep-dive)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

## 🎯 Project Overview

The pipeline performs the following key operations:

1. **3D Object Rendering**: Converts 3D objects into 8 different 2D views
2. **Multi-Caption Generation**: Generates 5 captions per view using OpenAI's GPT-4 Vision
3. **Intelligent Selection**: Selects the best caption per view using CLIP similarity scoring
4. **Consolidation**: Combines multi-view captions into a final comprehensive description

This approach enables detailed analysis of 3D parts for manufacturing, quality control, and documentation purposes.

## 📁 Project Structure

```
partFeatureAnalysisVLM/
├── 3d2txt/                          # Main pipeline directory
│   ├── 3D2Img_blender.py           # 3D to 2D rendering script
│   ├── img2txt_openai.py           # Image captioning with OpenAI
│   ├── select_best_caption_CLIP.py # CLIP-based caption selection
│   ├── summarize_captions_openai.py # Multi-view caption consolidation
│   └── example_material/           # Sample data and outputs
│       ├── Cap3D_captions/         # Generated captions (pickle files)
│       ├── Cap3D_imgs/            # Rendered 2D images (8 views)
│       ├── glbs/                  # Input 3D models (.glb files)
│       └── example_object_path.pkl # Object path mapping
├── pyproject.toml                  # Project dependencies
├── uv.lock                        # Dependency lock file
└── README.md                      # This file
```

## 🔄 Workflow

### Stage 1: 3D Rendering (`3D2Img_blender.py`)
- **Input**: 3D models (.glb files)
- **Process**: 
  - Loads 3D objects into Blender
  - Normalizes and centers objects
  - Renders 8 views from different angles
  - Saves images and camera matrices
- **Output**: 2D images for each view angle

### Stage 2: Image Captioning (`img2txt_openai.py`)
- **Input**: Rendered 2D images
- **Process**:
  - Encodes images to base64
  - Sends to OpenAI GPT-4 Vision API
  - Generates 5 captions per image
- **Output**: Pickle files with captions for each view

### Stage 3: Caption Selection (`select_best_caption_CLIP.py`)
- **Input**: Multiple captions per image
- **Process**:
  - Uses CLIP model to compute image-text similarity
  - Selects caption with highest similarity score
  - Combines best captions from all 8 views
- **Output**: CSV with selected captions per object

### Stage 4: Multi-View Consolidation (`summarize_captions_openai.py`)
- **Input**: Selected captions from all views
- **Process**:
  - Sends combined captions to OpenAI
  - Generates concise, unified description
- **Output**: Final summarized captions

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- Blender 3.6+
- NVIDIA GPU (recommended for rendering)
- OpenAI API key

### 1. Install Blender
```bash
# Download from https://www.blender.org/download/
# Install Pillow in Blender's Python environment
C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin> .\python.exe -m pip install Pillow
```

### 2. Install Python Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Install CLIP
```bash
uv pip install git+https://github.com/openai/CLIP.git
```

### 4. Configure OpenAI API
```bash
# Create .env file in your project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 💻 Usage

### 1. Prepare 3D Models
Place your `.glb` files in `3d2txt/example_material/glbs/`

### 2. Run the Complete Pipeline

```bash
# Stage 1: Render 3D objects to 2D images
cd 3d2txt
blender -b -P 3D2Img_blender.py --

# Stage 2: Generate captions for each image
python img2txt_openai.py

# Stage 3: Select best captions using CLIP
python select_best_caption_CLIP.py

# Stage 4: Summarize multi-view captions
python summarize_captions_openai.py \
  --input_csv example_material/Cap3D_captions/Cap3d_captions_final.csv \
  --output_csv example_material/Cap3D_captions/summarize.csv
```

## 🔬 Technical Deep Dive

### For Developers New to LLMs

This section explains the core concepts and configurations used in this project.

#### 1. Vision-Language Models (VLMs)

**What are VLMs?**
Vision-Language Models are AI systems that can understand both images and text. They bridge the gap between visual and textual understanding, enabling tasks like image captioning, visual question answering, and cross-modal retrieval.

**Key Components in This Project:**
- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model
- **GPT-4 Vision**: OpenAI's multimodal model for image understanding

#### 2. Core Configurations Explained

##### Blender Rendering Configuration (`3D2Img_blender.py`)

```python
# Render engine setup
bpy.context.scene.render.engine = 'CYCLES'  # Uses ray tracing for realistic rendering
bpy.context.scene.cycles.samples = 16       # Number of light samples (higher = better quality, slower)

# GPU acceleration
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'     # Use GPU for faster rendering
```

**Why These Settings Matter:**
- **CYCLES**: Provides photorealistic rendering with accurate lighting
- **GPU Rendering**: Significantly faster than CPU rendering
- **Sample Count**: Balances quality vs. speed (16 samples = good quality for ML)

##### OpenAI API Configuration (`img2txt_openai.py`)

```python
def get_openai_caption(image, model="gpt-4o", use_qa=True):
    # Image encoding
    base64_image = pil_image_to_base64(image)
    
    # Prompt engineering
    if use_qa:
        prompt = "Describe the structure, geometry, and detailed shape features of the main object in this image."
    else:
        prompt = "Describe the main object in this image."
    
    # API call configuration
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at generating concise and accurate captions for images."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=100,  # Limit response length
        n=5,            # Generate 5 different captions
    )
```

**Key Configuration Parameters:**
- **`max_tokens=100`**: Limits response length for consistency
- **`n=5`**: Generates multiple captions for selection
- **System prompt**: Guides the model's behavior
- **Base64 encoding**: Required format for image transmission

##### CLIP Model Configuration (`select_best_caption_CLIP.py`)

```python
# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Similarity computation
cos = CosineSimilarity(dim=1, eps=1e-6)

# Feature extraction
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)
scores = cos(image_features, text_features)
```

**CLIP Architecture Explained:**
- **ViT-B/32**: Vision Transformer with 32x32 patch size (good balance of speed/accuracy)
- **Cosine Similarity**: Measures similarity between image and text features
- **No gradients**: Inference-only mode for efficiency

#### 3. Data Flow and File Formats

##### Input Data Structure
```
glbs/
├── object_id_1.glb    # 3D model file
├── object_id_2.glb
└── ...

example_object_path.pkl  # Maps object IDs to file paths
```

##### Intermediate Data Structure
```
Cap3D_imgs/
├── Cap3D_imgs_view0/   # 8 different camera angles
├── Cap3D_imgs_view1/
├── ...
└── Cap3D_imgs_view7/

Cap3D_captions/
├── Cap3D_captions_view0.pkl  # Captions for each view
├── Cap3D_captions_view1.pkl
├── ...
└── Cap3D_captions_view7.pkl
```

##### Output Data Structure
```
Cap3D_captions/
├── Cap3d_captions_final.csv  # Selected captions per object
└── summarize.csv             # Final consolidated descriptions
```

#### 4. Error Handling and Robustness

##### API Error Handling
```python
def get_openai_caption(image, model="gpt-4o", use_qa=True):
    try:
        response = client.chat.completions.create(...)
        captions = [choice.message.content for choice in response.choices]
        return captions
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ["API error"]  # Graceful fallback
```

##### File Processing Robustness
```python
# Checkpoint saving during processing
if ct < 10 or (ct % 100 == 0 and ct < 1000) or (ct % 1000 == 0 and ct < 10000) or ct % 10000 == 0:
    with open(outfilename, 'wb') as f:
        pkl.dump(all_output, f)
```

#### 5. Performance Optimization

##### GPU Utilization
- **Blender**: CUDA rendering for faster image generation
- **CLIP**: GPU inference for similarity computation
- **Batch Processing**: Processes multiple images efficiently

##### Memory Management
- **Incremental Saving**: Saves progress periodically
- **No Gradient Storage**: Uses `torch.no_grad()` for inference
- **Image Preprocessing**: Resizes and normalizes images for CLIP

#### 6. Prompt Engineering

The project uses carefully crafted prompts to guide the AI models:

```python
# For detailed feature analysis
prompt = "Describe the structure, geometry, and detailed shape features of the main object in this image."

# For summarization
prompt = (
    "Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. "
    "Avoid describing background, surface, and posture. The caption should be:"
)
```

**Prompt Design Principles:**
- **Specificity**: Clear instructions about what to describe
- **Constraints**: Limits on what to avoid (background, posture)
- **Consistency**: Similar prompts across different views

## ⚙️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Parameters
- **GPT-4 Vision**: `gpt-4o` or `gpt-4-vision-preview`
- **CLIP Model**: `ViT-B/32` (default)
- **Rendering**: 16 samples, GPU acceleration
- **Caption Count**: 5 per image, 8 views per object

### File Paths
All paths are configurable via command-line arguments:
- `--parent_dir`: Base directory for data
- `--object_path_pkl`: Path to object mapping file
- `--input_csv`: Input CSV for summarization
- `--output_csv`: Output CSV for results

## 🎯 Next Steps

- **Injection Molding Integration**: Enhance prompts to include manufacturing-specific features
- **Quality Metrics**: Add evaluation metrics for caption quality
- **Batch Processing**: Optimize for large-scale processing
- **Web Interface**: Develop GUI for easier interaction

## 🙏 Acknowledgements

This project extends the [Cap3D](https://github.com/crockwell/Cap3D) framework, adapting it for manufacturing and part analysis applications.

---

**Note**: This pipeline requires OpenAI API credits for caption generation and summarization. Ensure you have sufficient credits before processing large datasets.

