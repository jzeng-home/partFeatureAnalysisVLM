import torch
from PIL import Image
import glob
import pickle as pkl
from tqdm import tqdm
import os
import argparse
import openai
import base64
from io import BytesIO
from dotenv import load_dotenv

load_dotenv(dotenv_path='D:/MyProjects/.env', override=True)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("API key loaded from .env")
else:
    print("API key NOT loaded from .env")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type = str, \
        default='D:/MyProjects/partFeatureAnalysisVLM/3d2txt/example_material/')
    parser.add_argument("--openai_api_key", type = str, default=api_key)
    parser.add_argument("--use_qa", action="store_false")
    parser.add_argument("--model", type=str, default="gpt-4o", \
        choices=["gpt-4o", "gpt-4-vision-preview"])
    return parser.parse_args()

client = openai.OpenAI(api_key=api_key)

def pil_image_to_base64(image):
    """Encodes a local image file into base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_openai_caption(image, model="gpt-4o", use_qa=True):

    # 1. Encode the image
    base64_image = pil_image_to_base64(image)
    # 2. Define the prompt
    if use_qa: # feature detection and description
        prompt = "Describe the structure, geometry, and detailed shape features of the main object in this image."
    else:
        prompt = "Describe the main object in this image."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at generating concise and accurate captions for images."},
                {
                    "role": "user",
                    "content": [
                        # The text prompt
                        {"type": "text", "text": prompt},
                        # The image
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=100,
            n=5,
        )
        captions = [choice.message.content for choice in response.choices]
        return captions
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ["API error"]


def img2txt(view_number):
    args = parse_args()
    all_output = {}
    outfilename = f'{args.parent_dir}/Cap3D_captions/Cap3D_captions_view{view_number}.pkl'
    infolder = f'{args.parent_dir}/Cap3D_imgs/Cap3D_imgs_view{view_number}/*.png'
    if os.path.exists(outfilename):
        with open(outfilename, 'rb') as f:
            all_output = pkl.load(f)
    print("number of annotations so far",len(all_output))
    all_files = glob.glob(infolder)
    all_imgs = [x for x in all_files if ".png" in x.split("_")[-1]]
    print("len of .png", len(all_imgs))
    all_imgs = [x for x in all_imgs if x not in all_output]
    print("len of new", len(all_imgs))
    ct = 0
    for filename in tqdm(all_imgs):
        if os.path.exists(outfilename):
            if os.path.basename(filename).split('.')[0] in all_output.keys():
                continue
        try:
            raw_image = Image.open(filename).convert("RGB")
        except:
            print("file not work skipping", filename)
            continue
        captions = get_openai_caption(raw_image, model=args.model, use_qa=args.use_qa)
        all_output[os.path.basename(filename).split('.')[0]] = captions
        if ct < 10 or (ct % 100 == 0 and ct < 1000) or (ct % 1000 == 0 and ct < 10000) or ct % 10000 == 0:
            print(filename)
            print(captions)
            with open(outfilename, 'wb') as f:
                pkl.dump(all_output, f)
        ct += 1
    with open(outfilename, 'wb') as f:
        pkl.dump(all_output, f)

if __name__ == "__main__":

    for i in range(8):
        img2txt(view_number=i)
