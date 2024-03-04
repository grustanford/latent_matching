# Semantic matching model
Semantic matching model aims to isolate semantically driven components from human visual search behavior. Our implementation leverages CLIP in a zero-shot manner to generate semantic similarity maps.

![](assets/readme/maps.png)

## Model usage
```python
from PIL import Image
import requests
from transformers import CLIPProcessor
from smm import SemanticMatchingModel

model = SemanticMatchingModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/train2017/000000341741.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["laptop", "chair"], 
    images=image, return_tensors="pt", padding=True
)

model.set_prototype(processor)  # set prototype target vector
outputs = model.get_maps(**inputs)  # [n_layers][n_images,h,w,n_texts] matching maps
```

## Model evaluation
We evaluated our model using human visual search behavior, COCO-SEARCH18 dataset. We report sAUC, uAUC, and NSS scores compared to other models, such as DeepGaze-IIE and Itti-Koch model.

```Makefile
make init # install dependencies
make generate-predictions # generate predictions
make evaluate-predictions # evaluate predictions
```

When you run `make generate-predictions`, it will generate predictions for COCO-SEARCH18 dataset. The predictions will be saved under `data/processed/maps/`, with the structure `{image_name}.pkl`. 

When you run `make evaluate-predictions`, it will evaluate the predictions and save the results under `data/processed/scores/`, with the structure `{model_name}.pkl`.


## Notebooks
To replicate the figures, visit notebooks/ and run the corresponding jupyter notebook.

## References
- [CLIP](https://arxiv.org/abs/2103.00020)
- [COCO-SEARCH18](https://sites.google.com/view/cocosearch/)
- [DeepGaze-IIE](https://arxiv.org/abs/2105.12441)