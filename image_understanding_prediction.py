from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES, BIOMED_HIERARCHY
import numpy as np
import os


from inference_utils.inference import interactive_infer_image

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

# text prompts querying objects in the image. Multiple ones can be provided.
prompts = BIOMED_CLASSES + list(BIOMED_HIERARCHY['CT'].keys()) + sum(list(BIOMED_HIERARCHY['CT'].values()), []) + \
          ["intracranial hemorrhage", "pulmonary embolism", "fracture", "cardiomegaly", "atelectasis"] + \
          ["pleural effusion", "pneumothorax", "pneumonia", "mass", "opacity"]
prompts = list(set(prompts))
print("prompts: ", prompts)

# Load image and run inference
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image_dir = '/local2/shared_data/VQA-RAD/images'
image_files = sorted(list(os.listdir(image_dir)))
full_image_file_paths = [os.path.join(image_dir, image_file) for image_file in image_files]
for image_path in full_image_file_paths:
    image = Image.open(image_path, formats=['jpeg'])
    image = image.convert('RGB')

    print("Running inference")
    pred_mask = interactive_infer_image(model, image, prompts)
    print(pred_mask.shape)
    print("Finished inference")
