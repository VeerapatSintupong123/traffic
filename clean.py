import os
import cv2 as cv
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)

def extract_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

def extract_image_features(image_paths, model_resnet, transform):
    image_features = {}
    for image_path in image_paths:
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model_resnet(image_tensor).squeeze().cpu().numpy() 

        image_features[image_path] = features
    return image_features

def create_similarity_matrix(image_features):
    image_names = list(image_features.keys())
    similarity_matrix = [
        [cosine_similarity(image_features[name1], image_features[name2]) for name2 in image_names]
        for name1 in image_names
    ]
    return pd.DataFrame(similarity_matrix, index=image_names, columns=image_names)

def remove_similar_images(similarity_df, dict_map, threshold=0.9):
    delete_image = set()
    for name in similarity_df.columns:
        if name in delete_image: continue
        similar_images = similarity_df[name].loc[similarity_df[name] > threshold].index.tolist()
        similar_images.remove(name)
        delete_image.update(similar_images)

    for name in delete_image:
        print(name)
        dict_map = [item for item in dict_map if item["path"] != name] 
        os.remove(name)

    return dict_map

def process_images(dict_map ,model_resnet, transform):
    image_paths = [map["path"] for map in dict_map]

    image_features = extract_image_features(image_paths, model_resnet, transform)
    similarity_df = create_similarity_matrix(image_features)

    result = remove_similar_images(similarity_df, dict_map)
    return result