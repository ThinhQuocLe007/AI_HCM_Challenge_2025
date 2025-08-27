# embed_utils.py
import torch
import copy
from PIL import Image
import numpy as np 
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token


def embed_text(text: str, model, tokenizer, device, conv_template="qwen_1_5", normalize=True):
    """
    Embed a text query using LLaVE's encode_multimodal_embeddings.
    Returns a (D,) float32 normalized numpy array.
    """

    # Prepare conversation prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], "\n")
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Forward
    with torch.no_grad():
        vec = model.encode_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # shape: (1, D)

    # Convert to NumPy
    vec = vec.squeeze(0).float().cpu().numpy()  # shape: (D,)
    if normalize:
        vec /= (np.linalg.norm(vec) + 1e-12)
    return vec


def embed_frame(image_path, model, tokenizer, image_processor, device, normalize=True):
    """
    Encode a single image using LLaVE.
    Returns: (1, D) numpy array
    """
    # Use a generic prompt: "Represent the given image."
    prompt = DEFAULT_IMAGE_TOKEN + " Describe the image in terms of what is happening, who is present, what objects are visible, and where the scene takes place."
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "\n")
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    dtype = torch.float16 if device == "cuda" else torch.float32
    image_tensor = [img.to(dtype=dtype, device=device) for img in image_tensor]
    image_sizes = [image.size]

    with torch.no_grad():
        emb = model.encode_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=image_sizes
        )
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()


