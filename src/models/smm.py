"""generate maps for semantic matching model (using CLIP)"""
import torch
from typing import Optional
from transformers import CLIPModel
from transformers.models.clip.configuration_clip import CLIPConfig

AVAILABLE_MODELS = [
    'ViT-B/32', 'ViT-L/14'
]

TARGETS_COCOSEARCH18 = {
    'bottle'       : 'a photo of a bottle',
    'bowl'         : 'a photo of a bowl',
    'car'          : 'a photo of a car',
    'chair'        : 'a photo of a chair',
    'clock'        : 'a photo of a clock',
    'cup'          : 'a photo of a cup',
    'fork'         : 'a photo of a fork',
    'keyboard'     : 'a photo of a keyboard',
    'knife'        : 'a photo of a knife',
    'laptop'       : 'a photo of a laptop',
    'microwave'    : 'a photo of a microwave',
    'mouse'        : 'a photo of a mouse',
    'oven'         : 'a photo of an oven',
    'potted plant' : 'a photo of a potted plant',
    'sink'         : 'a photo of a sink',
    'stop sign'    : 'a photo of a stop sign',
    'toilet'       : 'a photo of a toilet',
    'tv'           : 'a photo of a tv'
}

SIZE_CLIP = [16,16]
X_MAPS = slice(3,13)
Y_MAPS = slice(0,16)

class SemanticMatchingModel(CLIPModel):
    
    def __init__(self, config:CLIPConfig, targets=TARGETS_COCOSEARCH18):
        super().__init__(config)
        self.prototype = torch.zeros((1,self.projection_dim))
        self.targets   = targets
        self.resize    = True
        
    def set_prototype(self, processor):
        """set a target prototype representation"""        
        target_embeds = processor.tokenizer(text=list(self.targets.values()), return_tensors="pt", padding=True)
        target_embeds = self.text_model(**target_embeds.to(self.device))
        target_embeds = self.text_projection(target_embeds[1])
        target_embeds = target_embeds / target_embeds.norm(p=2, dim=-1, keepdim=True)
        prototype = torch.mean(target_embeds, dim=0, keepdims=True)
        self.prototype = prototype

    def get_maps(self,
                 input_ids: Optional[torch.LongTensor] = None,
                 pixel_values: Optional[torch.FloatTensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                ):
        """
        Returns: matching maps for each layer
        
        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor
        >>> import SemanticMatchingModel

        >>> model = SemanticMatchingModel.from_pretrained("openai/clip-vit-large-patch14")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        >>> url = "http://images.cocodataset.org/train2017/000000304815.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a keyboard", "a photo of a laptop", "a photo of a mouse"], 
        ...     images=image, return_tensors="pt", padding=True
        ... )

        >>> model.set_prototype(processor)  # set prototype target vector
        >>> outputs = model.get_maps(**inputs)  # [n_layers][n_images,height,width,n_texts] matching maps
        ```"""

        # text encoder
        text_outputs = self.text_model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            position_ids   = position_ids
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # image encoder
        image_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        # image-text matching
        image_embeds = [] # matching maps [N_Layer]
        for h in image_outputs['hidden_states']:
            image_embed = h[:, 1:, :] # take out the classification token
            image_embed = self.visual_projection(image_embed) # projection onto image-text space
            image_embed = image_embed @ (text_embeds - self.prototype).T # matching
            image_embed = torch.sigmoid(-image_embed) # nonlinearity

            if self.resize:
                image_embed = image_embed.reshape( (-1, *SIZE_CLIP, len(text_embeds)) )
                image_embed = image_embed[:, X_MAPS,Y_MAPS, :]

            image_embeds.append(image_embed)

        return image_embeds