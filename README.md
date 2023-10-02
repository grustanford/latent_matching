# Semantic matching model

Semantic matching model aims to isolate semantically driven components from human visual search behavior. Our implementation leverages [CLIP](https://arxiv.org/abs/2103.00020) in a zero-shot manner to generate semantic maps.

![](assets/readme/maps.pdf)

## Model usage

```python
from PIL import Image
import requests
from transformers import CLIPProcessor
from smm import SemanticMatchingModel

model = SemanticMatchingModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/train2017/000000304815.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a keyboard", "a photo of a laptop", "a photo of a mouse"], 
    images=image, return_tensors="pt", padding=True
)

model.set_prototype(processor)  # set prototype target vector
outputs = model.get_maps(**inputs)  # [n_layers][n_images,h,w,n_texts] matching maps
```

## Setup instructions

### Notebooks
To replicate the figures, visit notebooks/ and run the corresponding jupyter notebook. We believe each notebook is self-explanatory.