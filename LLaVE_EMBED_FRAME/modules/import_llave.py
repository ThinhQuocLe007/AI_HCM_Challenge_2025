# import_llave.py
import torch
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def load_llave_model():
    pretrained = "zhibinlan/LLaVE-0.5B"
    model_name = "llava_qwen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, model_name, device_map="auto"
    )
    model.eval()
    return tokenizer, model, image_processor, device
