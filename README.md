# CLIP-Inference

## Install requirements
```
pip install -r requirements.txt
```

## Call Inference
```python
from CLIPInference import main

from PIL import Image

prompt = "cat"
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
similarity_scores = embeddings.custom_embedding(prompt, image)
```