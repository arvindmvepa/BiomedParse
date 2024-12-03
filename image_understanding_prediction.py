from PIL import Image, ImageDraw, ImageFont
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES, BIOMED_HIERARCHY
import numpy as np
import os
import random
import cv2

from inference_utils.inference import interactive_infer_image

# Load options and initialize distributed settings
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
pretrained_pth = 'pretrained/biomed_parse.pt'
pretrained_pth = 'hf_hub:microsoft/BiomedParse'

print("Loading model from pretrained weights")
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
print("Loaded model")

with torch.no_grad():
    print("Generate text embeddings")
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    print("Generated text embeddings")

# Text prompts querying objects in the image. Multiple ones can be provided.
"""
prompts = BIOMED_CLASSES + list(BIOMED_HIERARCHY['CT'].keys()) + sum(list(BIOMED_HIERARCHY['CT'].values()), []) + \
          ["intracranial hemorrhage", "pulmonary embolism", "fracture", "cardiomegaly", "atelectasis"] + \
          ["pleural effusion", "pneumothorax", "pneumonia", "mass", "nodule"]
prompts = ["intracranial hemorrhage", "pulmonary embolism", "cardiomegaly", "atelectasis"] + \
          ["pleural effusion", "pneumothorax", "pneumonia", "mass", "opacity"]
"""
#prompts = ['tumor', 'nodule', 'opacity']
#prompts = ['tumor', 'nodule', 'COVID-19 infection']
prompts = ['tumor']
prompts = list(set(prompts))
if "other" in prompts:
    prompts.remove("other")
if "panreas" in prompts:
    prompts.remove("panreas")
print("prompts:", prompts)
print("Number of prompts:", len(prompts))

# Create an output directory if it doesn't exist
output_folder = 'output_images4'
os.makedirs(output_folder, exist_ok=True)

# Map prompts to indices
prompt_list = prompts
prompt_to_index = {prompt: idx for idx, prompt in enumerate(prompt_list)}

# Assign random colors to each prompt
color_map = {}
random.seed(42)  # For reproducibility
for prompt in prompt_list:
    color_map[prompt] = tuple(random.randint(0, 255) for _ in range(3))

# Load image and run inference
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.

#image_dir = '/local2/shared_data/VQA-RAD/images'
#image_files = sorted(list(os.listdir(image_dir)))
#full_image_file_paths = [os.path.join(image_dir, image_file) for image_file in image_files]
full_image_file_paths = ["/local2/amvepa91/BiomedParse/biomedparse_datasets/BiomedParseData-Demo/demo/02_CT_lung.png"]

for image_path in full_image_file_paths:
    image = Image.open(image_path, formats=[image_path[-3:]])
    image = image.convert('RGB')

    print("Running inference on", image_path)
    pred_mask = interactive_infer_image(model, image, prompts)
    print("Predicted mask shape:", pred_mask.shape)
    print("Finished inference")

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    pred_mask = np.where(pred_mask > 0.5, 1, 0)
    print("np.unique(pred_mask):", np.unique(pred_mask))

    N, H, W = pred_mask.shape
    pred_mask = pred_mask.transpose(1, 2, 0)  # Now pred_mask.shape == (H, W, N)
    print("Transposed pred_mask shape:", pred_mask.shape)

    # Verify that N matches the number of prompts
    if N != len(prompt_list):
        print(f"Warning: Number of channels in pred_mask ({N}) does not match number of prompts ({len(prompt_list)}).")
        # Adjust N and prompt_list accordingly
        min_N = min(N, len(prompt_list))
        pred_mask = pred_mask[:, :, :min_N]
        prompt_list = prompt_list[:min_N]
        print(f"Adjusted pred_mask and prompt_list to have {min_N} prompts.")

    # Resize pred_mask to match the original image size if necessary
    rgb_image = np.array(image)
    print("rgb_image shape:", rgb_image.shape)
    print("pred_mask shape before resizing:", pred_mask.shape)

    if pred_mask.shape[0] != rgb_image.shape[0] or pred_mask.shape[1] != rgb_image.shape[1]:
        # Resize pred_mask
        pred_mask_resized = np.zeros((rgb_image.shape[0], rgb_image.shape[1], pred_mask.shape[2]), dtype=pred_mask.dtype)
        for i in range(pred_mask.shape[2]):
            mask = Image.fromarray(pred_mask[:, :, i].astype(np.uint8))
            mask = mask.resize((rgb_image.shape[1], rgb_image.shape[0]), resample=Image.NEAREST)
            pred_mask_resized[:, :, i] = np.array(mask)
        pred_mask = pred_mask_resized
        print("Resized pred_mask to match rgb_image shape.")

    # Create a copy of the original image to draw contours
    contour_image = rgb_image.copy()

    # For each prompt/mask, extract contours and draw them
    for idx in range(pred_mask.shape[2]):
        mask = pred_mask[:, :, idx].astype(np.uint8)
        prompt = prompt_list[idx]
        color = color_map[prompt]
        # Find contours using OpenCV
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the image
        cv2.drawContours(contour_image, contours, -1, color, thickness=2)

    # Generate the legend entries
    legend_entries = []
    for idx, prompt in enumerate(prompt_list):
        legend_entries.append((prompt, color_map[prompt]))

    # Create the legend image
    legend_height = 20 * len(legend_entries) + 10
    legend_width = 250
    legend_image = Image.new('RGB', (legend_width, legend_height), color='white')
    draw = ImageDraw.Draw(legend_image)

    # Load a font (adjust the path to the font file if necessary)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for idx, (label_name, color) in enumerate(legend_entries):
        y = 10 + idx * 20
        # Draw color rectangle
        draw.rectangle([10, y, 30, y + 10], fill=color)
        # Draw text
        draw.text((35, y), label_name, fill='black', font=font)

    # Combine the contour image and the legend
    contour_pil_image = Image.fromarray(contour_image)
    contour_width, contour_height = contour_pil_image.size

    # Create a new image that can hold both the contour image and the legend
    total_width = contour_width + legend_width
    total_height = max(contour_height, legend_height)
    combined_image = Image.new('RGB', (total_width, total_height), color='white')

    # Paste the contour image and the legend into the combined image
    combined_image.paste(contour_pil_image, (0, 0))
    combined_image.paste(legend_image, (contour_width, 0))

    # Save the combined image
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_contours3.jpg'
    output_path = os.path.join(output_folder, output_filename)
    combined_image.save(output_path, 'JPEG')
    print(f"Saved output image to {output_path}")