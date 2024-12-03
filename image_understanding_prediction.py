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
prompts = BIOMED_CLASSES + list(BIOMED_HIERARCHY['CT'].keys()) + sum(list(BIOMED_HIERARCHY['CT'].values()), []) + \
          ["intracranial hemorrhage", "pulmonary embolism", "fracture", "cardiomegaly", "atelectasis"] + \
          ["pleural effusion", "pneumothorax", "pneumonia", "mass", "opacity"]
prompts = list(set(prompts))
prompts.remove("other")
prompts.remove("panreas")
print("prompts:", prompts)
print("Number of prompts:", len(prompts))

# Create an output directory if it doesn't exist
output_folder = 'output_images'
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
image_dir = '/local2/shared_data/VQA-RAD/images'
image_files = sorted(list(os.listdir(image_dir)))
full_image_file_paths = [os.path.join(image_dir, image_file) for image_file in image_files]

for image_path in full_image_file_paths:
    image = Image.open(image_path)
    image = image.convert('RGB')

    print("Running inference on", image_path)
    pred_mask = interactive_infer_image(model, image, prompts)
    print("Predicted mask shape:", pred_mask.shape)
    print("Finished inference")

    # Assuming pred_mask is of shape (N, H, W), where N is the number of prompts
    # Transpose pred_mask to shape (H, W, N)
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    N, H, W = pred_mask.shape
    pred_mask = pred_mask.transpose(1, 2, 0)  # Now pred_mask.shape == (H, W, N)
    print("Transposed pred_mask shape:", pred_mask.shape)

    # Verify that N matches the number of prompts
    if N != len(prompt_list):
        print(f"Warning: Number of channels in pred_mask ({N}) does not match number of prompts ({len(prompt_list)}).")
        # Handle this situation appropriately
        min_N = min(N, len(prompt_list))
        pred_mask = pred_mask[:, :, :min_N]
        prompt_list = prompt_list[:min_N]
        print(f"Adjusted pred_mask and prompt_list to have {min_N} prompts.")

    # Reshape pred_mask for easier processing
    pred_mask_reshaped = pred_mask.reshape(-1, N)

    # For each pixel, get the active prompts
    active_prompts_array = [tuple(np.where(pixel > 0)[0]) for pixel in pred_mask_reshaped]

    # Map unique combinations to labels, assign label 0 to background (empty combination)
    unique_combinations = set(active_prompts_array)
    combination_to_label = {}
    label_counter = 1
    for combination in unique_combinations:
        if len(combination) == 0:
            combination_to_label[combination] = 0  # Background label
        else:
            combination_to_label[combination] = label_counter
            label_counter += 1

    # Create combined_mask where each unique combination has a unique label
    combined_mask_flat = np.array([combination_to_label[combination] for combination in active_prompts_array])
    combined_mask = combined_mask_flat.reshape(H, W)

    # Map labels to colors (including overlaps)
    label_to_color = {}
    for combination, label in combination_to_label.items():
        if label == 0:
            continue  # Skip background
        # Ensure indices are within the valid range
        if any(idx >= len(prompt_list) for idx in combination):
            print(f"Invalid index in combination {combination}. Skipping.")
            continue

        if len(combination) == 1:
            # Single prompt
            prompt = prompt_list[combination[0]]
            label_to_color[label] = color_map[prompt]
        else:
            # Overlapping prompts: average their colors
            colors = np.array([color_map[prompt_list[idx]] for idx in combination])
            if colors.size == 0:
                print(f"No colors found for combination {combination}. Skipping.")
                continue
            mixed_color = tuple(np.mean(colors, axis=0).astype(np.uint8))
            label_to_color[label] = mixed_color

    # Convert the original image to a NumPy array
    rgb_image = np.array(image)

    # Create an overlay image
    overlay = np.zeros_like(rgb_image)

    print(overlay.shape)
    print(len(label_to_color.keys()))
    print([color.shape for color in label_to_color.values()])

    # Apply colors to the overlay based on the combined_mask
    for label in np.unique(combined_mask):
        if label == 0:
            continue  # Skip background
        if label not in label_to_color:
            continue  # Skip labels that were skipped earlier
        mask = combined_mask == label
        overlay[mask] = label_to_color[label]

    # Blend the overlay with the original image
    alpha = 0.5  # Transparency factor
    blended = (rgb_image * (1 - alpha) + overlay * alpha).astype(np.uint8)

    # Generate the legend entries
    legend_entries = []
    for combination, label in combination_to_label.items():
        if label == 0:
            continue  # Skip background
        if label not in label_to_color:
            continue  # Skip labels that were skipped
        prompt_names = [prompt_list[idx] for idx in combination if idx < len(prompt_list)]
        if len(prompt_names) == 1:
            label_name = prompt_names[0]
        else:
            label_name = ' & '.join(prompt_names)
        legend_entries.append((label_name, label_to_color[label]))

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

    # Combine the blended image and the legend
    blended_image = Image.fromarray(blended)
    blended_width, blended_height = blended_image.size

    # Create a new image that can hold both the blended image and the legend
    total_width = blended_width + legend_width
    total_height = max(blended_height, legend_height)
    combined_image = Image.new('RGB', (total_width, total_height), color='white')

    # Paste the blended image and the legend into the combined image
    combined_image.paste(blended_image, (0, 0))
    combined_image.paste(legend_image, (blended_width, 0))

    # Save the combined image
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_overlay.jpg'
    output_path = os.path.join(output_folder, output_filename)
    combined_image.save(output_path, 'JPEG')
    print(f"Saved output image to {output_path}")
