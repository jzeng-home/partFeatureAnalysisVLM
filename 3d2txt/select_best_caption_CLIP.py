import pickle

from pathlib import Path

import os
import pickle
import torch
import clip
from PIL import Image
from torch.nn import CosineSimilarity
import csv
import argparse
from functools import reduce

# set up CLIP
cos = CosineSimilarity(dim=1, eps=1e-6)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

parent_dir='D:/MyProjects/partFeatureAnalysisVLM/3d2txt/example_material/'

caps = []
for i in range(8):
    caps.append(pickle.load(open(os.path.join(parent_dir,'Cap3D_captions','Cap3D_captions_view'+str(i)+'.pkl'), 'rb')))

names = []
for i in range(8):
    names.append(set([name.split('_')[0] for name in caps[i].keys()]))
object_ids = list(reduce(set.intersection, names))
#print(object_ids)

output_csv = open(os.path.join(parent_dir, 'Cap3D_captions', 'Cap3d_captions_final.csv'), 'w')
writer = csv.writer(output_csv)

for i, obj_id in enumerate(object_ids):

    obj_final_caption = ''

    for k in range(8):
        image_file_name = os.path.join(parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d'%k, obj_id + '_%d.png'%k)
        #print(image_file_name)
        image=preprocess(Image.open(image_file_name)).unsqueeze(0).to(device)
        
        obj_caption = caps[k][obj_id+'_%d'%k]
        #print("k=",k, "idx=", (obj_id+'_%d'%k) )
        #print(obj_caption)

        texts = clip.tokenize(obj_caption, truncate=True).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(texts)
        scores = cos(image_features, text_features)

        if k == 8-1:  # last view
            obj_final_caption += obj_caption[torch.argmax(scores)]
        else:
            obj_final_caption += obj_caption[torch.argmax(scores)] + ', '

    
    #print(i, obj_id, obj_final_caption)

    # write to csv
    writer.writerow([i, obj_id, obj_final_caption])
    if (i)% 1000 == 0:
        output_csv.flush()
        os.fsync(output_csv.fileno())
    
output_csv.close()


#helper 

def find_pkl_files(folder_path):
    """
    Finds all .pkl files in a given folder using pathlib.

    Args:
        folder_path (str or Path): The path to the folder to search.

    Returns:
        list: A list of Path objects for each .pkl file found.
              Returns an empty list if the folder doesn't exist.
    """
    # Create a Path object from the input string
    search_path = Path(folder_path)

    # Check if the folder actually exists
    if not search_path.is_dir():
        print(f"❌ Error: Folder not found at '{folder_path}'")
        return []

    # The glob() method finds all files matching the pattern.
    # The pattern '*.pkl' means any filename ending with .pkl.
    pkl_files = list(search_path.glob('*.pkl'))
    
    return pkl_files


def load_all_pickles_from_folder(folder_path):
    """
    Finds and loads all .pkl files from a folder into a dictionary.

    Args:
        folder_path (str or Path): The path to the folder.

    Returns:
        dict: A dictionary where keys are the filenames (without .pkl)
              and values are the loaded Python objects.
    """
    search_path = Path(folder_path)
    if not search_path.is_dir():
        print(f"❌ Error: Folder not found at '{folder_path}'")
        return {}

    all_loaded_data = {}
    
    print(f"🔍 Searching for .pkl files in '{search_path}'...")
    
    for file_path in search_path.glob('*.pkl'):
        print(f"  -> Found '{file_path.name}'. Loading...")
        try:
            with open(file_path, 'rb') as f:
                # Use file_path.stem to get the filename without the extension
                filename_key = file_path.stem 
                all_loaded_data[filename_key] = pickle.load(f)
        except pickle.UnpicklingError:
            print(f"     ⚠️  Warning: Could not unpickle '{file_path.name}'. File might be corrupted.")
        except Exception as e:
            print(f"     ❌  Error: An unexpected error occurred with '{file_path.name}': {e}")
            
    return all_loaded_data

def print_kv_from_data():
    folder = 'D:/MyProjects/partFeatureAnalysisVLM/3d2txt/example_material/Cap3D_captions'  
    pickle_files = find_pkl_files(folder)
    all_data = load_all_pickles_from_folder(folder)

    print("\n" + "="*40)
    if all_data:
        print("✅ All pickle files loaded successfully!")
        print("\nContents:")
        for name, data in all_data.items():
            print(f"\n--- From file '{name}.pkl' ---")
            print(f"Type: {type(data)}")
            print(f"image name: {data.keys()}")
            print(f"image caption: {data.values()}")
            exit(0)
    else:
        print("No data was loaded.")

