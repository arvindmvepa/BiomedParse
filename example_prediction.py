from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
import numpy as np

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

# Load image and run inference
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open('biomedparse_datasets/BiomedParseData-Demo/demo/02_CT_lung.png', formats=['png'])
image = image.convert('RGB')
# text prompts querying objects in the image. Multiple ones can be provided.
prompts = ['tumor']

# load ground truth mask
gt_masks = []
for prompt in prompts:
    gt_mask = Image.open(f"biomedparse_datasets/BiomedParseData-Demo/demo_mask/02_CT_lung_{prompt.replace(' ', '+')}.png", formats=['png'])
    gt_mask = 1*(np.array(gt_mask.convert('RGB'))[:,:,0] > 0)
    gt_masks.append(gt_mask)

print("Running inference")
pred_mask = interactive_infer_image(model, image, prompts)
print("Finished inference")

# prediction with ground truth mask
for i, pred in enumerate(pred_mask):
    gt = gt_masks[i]
    dice = (1*(pred>0.5) & gt).sum() * 2.0 / (1*(pred>0.5).sum() + gt.sum())
    print(f'Dice score for {prompts[i]}: {dice:.4f}')