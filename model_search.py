import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd 
import ftfy 

# --- Configuration ---
# FORCING CPU for stability and reliable model loading (avoids CUDA/GPU issues)
DEVICE = "cpu" 
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDINGS_FILE = "product_embeddings.npy"
METADATA_FILE = "product_metadata.csv" 
DATA_DIR = "data/product_images" 
# CRITICAL FIX: Set to the exact metadata file name from Kaggle
KAGGLE_METADATA_CSV = "styles.csv" 

# Load Model and Processor (Run only once to save time)
try:
    CLIP_MODEL = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    CLIP_MODEL = None
    CLIP_PROCESSOR = None

# --- Core Functions ---

def generate_embedding(image):
    """Generates a CLIP embedding (vector) for a single PIL image."""
    if CLIP_MODEL is None:
        return None 
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        image_features = CLIP_MODEL.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()


def build_and_save_index(image_directory):
    """
    1. Indexes all images based on the KAGGLE_METADATA_CSV file, limited to 30 items.
    2. Stores the embeddings and clean metadata.
    """
    if CLIP_MODEL is None:
        return 0, "Model Not Loaded"

    # --- Indexing Limit Configuration (CRITICAL FIX) ---
    MAX_ITEMS_TO_INDEX = 200 
    current_index_count = 0
    # ----------------------------------------------------

    # 1. Load Kaggle Metadata
    if not os.path.exists(KAGGLE_METADATA_CSV):
        return 0, f"Metadata file not found: {KAGGLE_METADATA_CSV}. Cannot build index."
    
    try:
        metadata_df = pd.read_csv(KAGGLE_METADATA_CSV, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        return 0, f"Error reading CSV file: {e}"

    
    embeddings = []
    clean_metadata = []
    
    # 2. Find All Images (Robust Search)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        # ** allows searching recursively inside subfolders of the image_directory
        image_paths.extend(glob.glob(os.path.join(image_directory, "**", ext), recursive=True))

    
    if not image_paths:
        return 0, f"No images found in the directory '{image_directory}'. Check file extensions."

    # 3. Iterate through Metadata and Process Images
    for index, row in metadata_df.iterrows():
        
        # --- Break condition to limit the number of items ---
        if current_index_count >= MAX_ITEMS_TO_INDEX:
            break
        # --------------------------------------------------

        # Find the image file using the ID from the CSV
        image_id = str(row['id'])
        found_path = None
        for path in image_paths:
            # Check if the path ends with the image ID + extension (case-insensitive check)
            if path.lower().endswith(f"{image_id}.jpg") or path.lower().endswith(f"{image_id}.png"):
                 found_path = path
                 break
        
        if not found_path:
             continue 

        try:
            # 4. Generate Embedding
            image = Image.open(found_path).convert("RGB")
            embedding = generate_embedding(image)
            
            if embedding is None:
                continue

            embeddings.append(embedding)
            
            # 5. Save Cleaned Metadata (Using real data from the CSV)
            clean_metadata.append({
                'id': row['id'], 
                # These column names are standard for the Fashion Product Images Dataset on Kaggle
                'name': row['productDisplayName'], 
                'category': row['masterCategory'], 
                'path': found_path 
            })
            
            current_index_count += 1 # Increment counter only upon successful processing
            
        except Exception as e:
            print(f"Could not process image {row['id']} at {found_path}: {e}")
            continue

    # 6. Final Save
    if not embeddings:
         return 0, "All images failed to process (0 embeddings generated). Check terminal for file corruption errors."

    np.save(EMBEDDINGS_FILE, np.array(embeddings))
    
    df_index = pd.DataFrame(clean_metadata)
    df_index.to_csv(METADATA_FILE, index=False)
    
    return len(embeddings), "Index Built Successfully"

def search_similar_products(query_image, k=3):
    """
    Performs the visual similarity search.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        return [], "Index not built. Run indexing first."

    # 1. Load the Index
    product_embeddings = np.load(EMBEDDINGS_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)
    
    # 2. Generate Query Embedding
    query_embedding = generate_embedding(query_image)
    if query_embedding is None:
         return [], "Model not ready."

    # Reshape for scikit-learn
    query_embedding = query_embedding.reshape(1, -1)
    
    # 3. Calculate Cosine Similarity (The core of the visual search)
    similarities = cosine_similarity(query_embedding, product_embeddings).flatten()
    
    # 4. Get the top K indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # 5. Retrieve Results
    results = []
    for i in top_indices:
        match = metadata_df.iloc[i].to_dict()
        match['similarity_score'] = round(float(similarities[i]), 4)
        results.append(match)
        
    return results, "Search Complete"