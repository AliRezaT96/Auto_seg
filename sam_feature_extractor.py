import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default="./example_dataset")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    dataset_path = args.dataset_path

    images_folder = os.path.join(dataset_path, "images")
    embeddings_folder = os.path.join(dataset_path, "embeddings")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)