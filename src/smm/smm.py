"""generate maps for semantic matching model (using CLIP)"""
import torch
import numpy as np
from PIL import Image
from typing import Optional
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers.tokenization_utils_base import BatchEncoding

TARGETS_COCOSEARCH18 = {
    'bottle'       : 'bottle',
    'bowl'         : 'bowl',
    'car'          : 'car',
    'chair'        : 'chair',
    'clock'        : 'clock',
    'cup'          : 'cup',
    'fork'         : 'fork',
    'keyboard'     : 'keyboard',
    'knife'        : 'knife',
    'laptop'       : 'laptop',
    'microwave'    : 'microwave',
    'mouse'        : 'mouse',
    'oven'         : 'oven',
    'potted plant' : 'potted plant',
    'sink'         : 'sink',
    'stop sign'    : 'stop sign',
    'toilet'       : 'toilet',
    'tv'           : 'tv'
}


# todo : currently, the slices are hardcoded for the coco_search18 dataset
# todo : slices and padding of images should be automatically determined
SLICES_COCOSEARCH18 = {
    # slice of the image to crop { clip_size: (h_slice, w_slice) }
    7  : ( slice( 1, 6 ), slice( 0, 7 ) ),
    16 : ( slice( 3,13 ), slice( 0,16 ) ),
}


class Encoding(BatchEncoding):

    def to(self, device):
        from transformers.utils import requires_backends, is_torch_device
        requires_backends(self, ["torch"])

        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            self.data = {k: [_v.to(device=device) for _v in v] if isinstance(v, list) else v.to(device=device) 
                         for k, v in self.data.items()}
        else:
            print(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")

        return self


class SemanticMatchingProcessor(CLIPProcessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, text=None, images=None, 
                 return_tensors=None, 
                 patch_size=14,
                 super_resolution=None,
                 n_shifts=(1,1),
                 **kwargs):
        """
        Overriding the `__call__` method of CLIPProcessor.

        Args:
            text (str or List[str]):
                The text to be encoded. Can be a string, a list of strings (tokenized string using the `self.tokenizer`).
            images (str or List[str]):
                The image to be encoded. Can be a string representing the path to an image, a list of strings
                representing paths to images or a list of numpy.ndarray.
            return_tensors (str, optional):
                If set, will return tensors instead of list of python integers.
                Acceptable values are 'tf' and 'pt' (default: None).
            patch_size (int, optional):
                The patch size to use when encoding the image.
            super_resolution (float, optional):
                The super resolution factor. 1 means full resolution compared to the patch size.
        """
 
        if text is None or images is None:
            raise ValueError("You have to specify both text and images.")

        if isinstance(images, Image.Image):
            images = [images]

        if isinstance(text, str):
            raise ValueError("The text should be a list of strings.")

        elif isinstance(text[0], str):
            text = [text] * len(images)
        
        elif len(text) != len(images):
            raise ValueError("The number of text and images should be the same.")
        
        # text encoding
        text = [self.tokenizer(t, return_tensors=return_tensors, **kwargs) for t in text]
        encoding = {}
        for key in text[0].keys():
            encoding[key] = [t[key] for t in text]

        # image preprocessing
        if super_resolution is not None:
            if self.feature_extractor.do_resize:
                images = [self.feature_extractor.resize(np.array(image), self.feature_extractor.size) for image in images]
            
            if self.feature_extractor.do_center_crop:
                images = [self.feature_extractor.center_crop(image, self.feature_extractor.crop_size) for image in images]

            step_size = int(1./super_resolution)
            images = [img for image in images for img in self.image_shift(image, patch_size, (step_size,step_size))]

        # image encoding
        image_features = self.feature_extractor(images, return_tensors=return_tensors, **kwargs)
        
        if super_resolution is not None:
            n_shifts = [int(patch_size*super_resolution)] * 2

        n_sub_images = np.prod(n_shifts, dtype=int)
        encoding["pixel_values"] = [image_features.pixel_values[i:i+n_sub_images] for i in range(0, len(images), n_sub_images)]
 
        return Encoding(encoding)
    

    def image_shift(self, image, patch_size, shift_sizes):
        """shift the image by shift_sizes (h,w) and return a list of images"""
        image = Image.fromarray(image)
        w, h  = image.size
        sh,sw = shift_sizes
        padded_image = Image.new("RGB", (w+patch_size, h+patch_size), (0,0,0))
        padded_image.paste(image, (patch_size//2, patch_size//2))

        images = []
        for i_w in range(0,patch_size,sw):
            for i_h in range(0,patch_size,sh):
                images.append( np.array(padded_image.crop((i_w, i_h, w+i_w, h+i_h ))) )
        return images
        

    def post_process(self, 
                     outputs,
                     patch_size=14,
                     super_resolution=None,
                     n_shifts=(1,1),
                     slices=None):
        """
        post-process the outputs by slicing and spatial registration. 

        Args:
            outputs (list):
                outputs to be post-processed. n_images-list of tensors of (n_subimages, n_targets, n_layers, ...)
            patch_size (int, optional):
                The patch size to use when encoding the image.
            super_resolution (float, optional):
                The super resolution factor. 1 means full resolution compared to the patch size.
            n_shifts (tuple, optional):
                The number of shifts to use when encoding the image.
            slices (tuple, optional):
                The slice of the image to crop (h_slice, w_slice)

        Returns:
            outputs (list):
                post-processed outputs. n_images-list of arrays of (n_targets, n_layers, ...)
        """
        if super_resolution is not None:
            n_shifts = [int(patch_size*super_resolution)] * 2
        
        for img, out in enumerate(outputs):

            out = out.detach().cpu().numpy()
            if slices is not None:
                out = out[..., slices[0], slices[1]]

            out = [
                np.stack([
                    self.maps_registration(out[:,t,l], n_shifts=n_shifts) for l in range(out.shape[2])
                ]) for t in range(out.shape[1])
            ]
            outputs[img] = np.stack(out)
        
        return outputs


    def maps_registration(self, maps, n_shifts):
        """spatial registration of maps, given n_shifts (h,w). expects maps in numpy format"""
        h,w = maps[0].shape
        register_h = int(h*n_shifts[0])
        register_w = int(w*n_shifts[1])
        
        map_registered = np.nan*np.zeros((register_h,register_w))
        for i_w in range(n_shifts[1]):
            for i_h in range(n_shifts[0]):
                idxw = np.arange(i_w,register_w,step=n_shifts[1])
                idxh = np.arange(i_h,register_h,step=n_shifts[0])
                idxw, idxh = np.meshgrid(idxw, idxh)
                ii = int(i_w*n_shifts[1] + i_h)
                map_registered[idxh.flatten(), idxw.flatten()] = maps[ii].flatten()

        return map_registered



class SemanticMatchingModel(CLIPModel):
    
    def __init__(self, config:CLIPConfig, 
                 targets=TARGETS_COCOSEARCH18,
                 slices=SLICES_COCOSEARCH18):
        super().__init__(config)
        self.origin    = torch.zeros((1,self.projection_dim))
        self.targets   = targets
        self.get_map_size()
        self.slices    = slices[self.size_map[0]]


    def set_origin(self, processor):
        """set the origin of target representations"""        
        target_embeds = processor.tokenizer(text=list(self.targets.values()), return_tensors="pt", padding=True)
        target_embeds = self.get_text_embeds(**target_embeds.to(self.device))
        origin = torch.mean(target_embeds, dim=0, keepdims=True)
        self.origin = origin


    def get_map_size(self):
        """get the number of patches"""
        im_sz = self.config.vision_config.image_size
        ph_sz = self.config.vision_config.patch_size
        self.size_map = [ int(im_sz/ph_sz), int(im_sz/ph_sz) ]


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
        """get matching maps [n_images,n_texts,n_layers,n_h,n_w]"""
        matches = []
        for h in image_embeds['hidden_states']:
            match = self.visual_projection(h) # projection onto image-text space
            match = match[:, 1:, :] # take out the classification token
            match = match @ (text_embeds - self.origin).T # matching
            match = torch.sigmoid(-match) # nonlinearity
            match = match.reshape( (-1, *self.size_map, len(text_embeds)) )
            matches.append(match)
        n_subimages = len(match)
        matches  = [torch.stack([match[i].permute(2,0,1) for match in matches], dim=1) for i in range(n_subimages)]
        matches  = torch.stack(matches)
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
        >>> from smm import SemanticMatchingModel, SemanticMatchingProcessor

        >>> model = SemanticMatchingModel.from_pretrained("openai/clip-vit-large-patch14")
        >>> processor = SemanticMatchingProcessor.from_pretrained("openai/clip-vit-large-patch14")
        >>> model.set_origin(processor)  # set target origin vector
        
        >>> url = "http://images.cocodataset.org/train2017/000000341741.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # basic resolution (16x16)
        >>> inputs = processor(
        ...     text=["laptop", "chair"],
        ...     images=image, return_tensors="pt", padding=True
        ... )
        >>> outputs = model.get_maps(**inputs)
        >>> outputs = processor.post_process(outputs)  # [n_pair][n_tgt][n_layer,h,w] matching maps

        >>> # higher resolution (224x224)
        >>> inputs = processor(
        ...     text=["laptop", "chair"],
        ...     images=image, return_tensors="pt", padding=True,
        ...     super_resolution=1.0
        ... )
        >>> outputs = model.get_maps(**inputs)
        >>> outputs = processor.post_process(outputs, super_resolution=1.)  # [n_pair][n_tgt][n_layer,h,w] matching maps
        ```"""

        if len( input_ids ) != len( pixel_values ):
            raise ValueError("The number of text and images should be the same.")
        
        if position_ids is not None: raise NotImplementedError

        # text encoder
        text_embeds = [
            self.get_text_embeds(
                input_ids      = ids,
                attention_mask = mask
            ) for ids, mask in zip(input_ids, attention_mask)
        ]

        # image encoder
        image_embeds = [ self.get_image_embeds(pv) for pv in pixel_values ]

        # image-text matching
        matches = [ self.get_matches(t,i) for t,i in zip(text_embeds, image_embeds) ]

        return matches