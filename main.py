import torch
import cv2
from PIL import Image

import open_clip
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel

models = {
    'clip': {
        'model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
        'processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        'padding': True
    },
    'siglip': {
        'model': AutoModel.from_pretrained("google/siglip-base-patch16-256-i18n"),
        'processor': AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n"),
        'padding': 'max_length'
    }, 'CLIP-ViT-L-14-DataComp.XL-s13B-b90K': {
        'model': open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'),
        'tokenizer': open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    }
}

CLIP_MODEL = 'CLIP-ViT-L-14-DataComp.XL-s13B-b90K'

def custom_embedding(text, images, model=CLIP_MODEL):
    model_obj = models.get(model)
    if model_obj is None:
        model_obj = models['clip']
    if model == 'clip':
        model = model_obj['model']
        processor = model_obj['processor']
        padding = model_obj['padding']
        
        inputs = processor(text=text, images=images, return_tensors="pt", padding=padding)
        outputs = model(**inputs)
        similarities = outputs.logits_per_text.softmax(dim=1).squeeze()
    elif model == 'CLIP-ViT-L-14-DataComp.XL-s13B-b90K':
        model, preprocess = model_obj['model']
        tokenizer = model_obj['tokenizer']
        text = tokenizer([text])
        images = preprocess_frames_for_clip(images, preprocess)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)

            # Normalize the features to unit length
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute the similarity scores (dot product of normalized features)
            similarities = text_features @ image_features.T
            similarities = similarities.squeeze(0)

    return similarities

def preprocess_frames_for_clip(frames, preprocess):
    images = []
    for frame in frames:
        image = convert_frame_for_clip(frame)
        image = preprocess(image).unsqueeze(0)
        images.append(image)
    return torch.cat(images, dim=0)

def convert_frame_for_clip(frame):
    """
    Convert a frame from OpenCV's BGR format to PIL's RGB format for CLIP processing.

    Args:
    frame (numpy.ndarray): The frame in BGR format.

    Returns:
    PIL.Image: The converted frame in RGB format.
    """
    if frame is not None:
        # Convert the color from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image
        return Image.fromarray(frame_rgb)
    else:
        return None
    
