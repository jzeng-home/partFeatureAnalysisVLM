---

# Part Feature Analysis / Labeling via VLM

This directory hosts our pipeline for analyzing and labeling 3D part features using vision-language models (VLMs). The process involves

1. **Rendering 3D objects into eight views**
2. **Generating five captions per view with OpenAI calls**
3. **Selecting one caption per view using CLIP**
4. **Consolidating a final caption from multi-view using OpenAI calls**

---

## Next Steps

- **Prompt with injection molding features to bind part analysis with process**

---

## Pipeline Overview

### 1. Rendering

- **Install Blender:**  
  [Download Blender](https://www.blender.org/download/)

- **Install Pillow library in Blender:**
  ```sh
  C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin> .\python.exe -m pip install Pillow
  ```

- **Inside the `3d2txt/` folder, run:**
  ```sh
  blender -b -P 3D2Img_blender.py --
  ```
  *Note: Both inputs and outputs are set to default in the script.*

### 2. Image → Text

- **Generate captions for each image:**
  ```sh
  python ./img2txt_openai.py
  ```

### 3. Caption Selection (CLIP)

- **Install CLIP:**
  ```sh
  uv pip install git+https://github.com/openai/CLIP.git
  ```

- **For each viewing angle, select the best out of 5 captions using CLIP.**
  Assemble the best of 8 viewing angles to form a prompt describing this object:
  ```sh
  python ./select_best_caption_CLIP.py
  ```

### 4. Multiple 2D Captions → 3D Caption

- **Summarize the captions using OpenAI:**
  ```sh
  python summarize_captions_openai.py \
    --input_csv 3d2txt/example_material/Cap3D_captions/Cap3d_captions_final.csv \
    --output_csv 3d2txt/example_material/Cap3D_captions/summarize.csv \
    --openai_api_key YOUR_KEY
  ```
  *You can also set your API key in a `.env` file as `OPENAI_API_KEY`.*


---

## Acknowledgement

**This is an extension of [Cap3D](https://github.com/crockwell/Cap3D).**

