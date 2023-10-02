"""generate maps for semantic matching model (using CLIP)"""
import torch
from typing import Optional
from transformers import CLIPModel
from transformers.models.clip.configuration_clip import CLIPConfig

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

SLICES_COCOSEARCH18 = {
    # slice of the image to crop { clip_size: (h_slice, w_slice) }
    7  : ( slice( 1, 6 ), slice( 0, 7 ) ),
    16 : ( slice( 3,13 ), slice( 0,16 ) ),
}


class SemanticMatchingModel(CLIPModel):
    
    def __init__(self, config:CLIPConfig, 
                 targets=TARGETS_COCOSEARCH18,
                 slices=SLICES_COCOSEARCH18):
        super().__init__(config)
        self.prototype = torch.zeros((1,self.projection_dim))
        self.targets   = targets
        self.resize    = True
        self.get_clip_size()
        self.h_slice   = slices[self.size_clip[0]][0]
        self.w_slice   = slices[self.size_clip[1]][1]


    def set_prototype(self, processor):
        """set a target prototype representation"""        
        target_embeds = processor.tokenizer(text=list(self.targets.values()), return_tensors="pt", padding=True)
        target_embeds = self.get_text_embeds(**target_embeds.to(self.device))
        prototype = torch.mean(target_embeds, dim=0, keepdims=True)
        self.prototype = prototype


    def get_clip_size(self):
        """get the number of patches in CLIP"""
        im_sz = self.config.vision_config.image_size
        ph_sz = self.config.vision_config.patch_size
        self.size_clip = [ int(im_sz/ph_sz), int(im_sz/ph_sz) ]


    def get_image_embeds(self, 
                         pixel_values: Optional[torch.FloatTensor] = None,
                         **kwargs
                         ):
        """get image embeddings"""
        image_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        return image_embeds


    def get_text_embeds(self,
                        input_ids: Optional[torch.LongTensor] = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        **kwargs
                        ):
        """get text embeddings"""
        text_outputs = self.text_model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            position_ids   = position_ids
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        return text_embeds


    def get_matches(self, text_embeds, image_embeds, **kwargs):
        """get matching maps"""
        matches = [] # matching maps [N_Layer]
        for h in image_embeds['hidden_states']:
            match = self.visual_projection(h) # projection onto image-text space
            match = match[:, 1:, :] # take out the classification token
            match = match @ (text_embeds - self.prototype).T # matching
            match = torch.sigmoid(-match) # nonlinearity
            if self.resize:
                match = match.reshape( (-1, *self.size_clip, len(text_embeds)) )
                match = match[:, self.h_slice, self.w_slice, :]
            matches.append(match)
        return matches


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
        text_embeds = self.get_text_embeds(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            position_ids   = position_ids
        )
        
        # image encoder
        image_embeds = self.get_image_embeds(
            pixel_values=pixel_values
        )
        
        # image-text matching
        matches = self.get_matches(text_embeds, image_embeds)
        
        return matches