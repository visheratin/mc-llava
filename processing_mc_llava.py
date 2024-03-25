import math
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    ImageProcessingMixin,
    ProcessorMixin,
    SiglipImageProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType


class MultiCropImageProcessor(ImageProcessingMixin):
    def __init__(self, model_name, max_crops=0, **kwargs):
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.crop_size = 384
        self.max_crops = max_crops

    def __call__(
        self,
        images: List[Image.Image],
        max_crops: int = -1,
    ):
        pixel_values = []
        coords = []
        if max_crops < 0:
            max_crops = self.max_crops
        crop_nums = []
        for image in images:
            crop_nums.append(self.crops_num(image, self.crop_size))
        mean_crop_num = int(sum(crop_nums) / len(crop_nums))
        if mean_crop_num > max_crops:
            mean_crop_num = max_crops
        crops_num = self.round_num_crops(mean_crop_num)
        for image in images:
            outputs, output_coords = self.process_image(image, crops_num)
            pixel_values.append(outputs)
            coords.append(output_coords)
        pixel_values = torch.stack(pixel_values)
        coords = torch.stack(coords)
        return pixel_values, coords

    def round_num_crops(self, n_crops: int):
        highly_composite_numbers = [
            1,
            2,
            4,
            6,
            12,
            24,
            36,
            48,
            60,
            120,
            180,
            240,
        ]
        for i in range(len(highly_composite_numbers) - 1, -1, -1):
            if highly_composite_numbers[i] <= n_crops:
                return highly_composite_numbers[i]

    def crops_num(self, image: Image.Image, crop_size: int):
        width, height = image.size
        x_steps = width // crop_size
        y_steps = height // crop_size
        return x_steps * y_steps

    def process_image(self, image: Image.Image, crops_num: int):
        whole_image_res = self.processor(image, return_tensors="pt").pixel_values
        whole_image_coords = torch.tensor([0.5, 0.5, 1.0, 1.0])
        outputs = []
        output_coords = []
        width, height = image.size
        aspect_ratio = width / height
        aspect_sum = aspect_ratio + 1
        small_crop_num = round(math.sqrt(crops_num / aspect_sum))
        numbers = [1, 2, 3, 4, 5, 6]
        for _, number in enumerate(numbers):
            if number >= small_crop_num and crops_num % number == 0:
                small_crop_num = number
                break
        large_crop_num = crops_num // small_crop_num
        if aspect_ratio > 1:
            x_steps = large_crop_num
            y_steps = small_crop_num
        else:
            x_steps = small_crop_num
            y_steps = large_crop_num
        if x_steps < 1:
            x_steps = 1
        if y_steps < 1:
            y_steps = 1
        if x_steps == 1 and y_steps == 1:
            return self.processor(
                image, return_tensors="pt"
            ).pixel_values, torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        x_crop_size = width // x_steps
        y_crop_size = height // y_steps
        x_coords = [[i * x_crop_size, (i + 1) * x_crop_size] for i in range(x_steps)]
        if x_coords[-1][1] != width:
            x_coords[-1][1] = width
        y_coords = [[i * y_crop_size, (i + 1) * y_crop_size] for i in range(y_steps)]
        if y_coords[-1][1] != height:
            y_coords[-1][1] = height
        for _, y_coord in enumerate(y_coords):
            for _, x_coord in enumerate(x_coords):
                crop = image.crop((x_coord[0], y_coord[0], x_coord[1], y_coord[1]))
                outputs.append(self.processor(crop, return_tensors="pt").pixel_values)
                output_coords.append(
                    torch.tensor(
                        [
                            (x_coord[0] + x_coord[1]) / 2 / width,
                            (y_coord[0] + y_coord[1]) / 2 / height,
                            (x_coord[1] - x_coord[0]) / width,
                            (y_coord[1] - y_coord[0]) / height,
                        ]
                    )
                )
        outputs.append(whole_image_res)
        output_coords.append(whole_image_coords)
        outputs = torch.cat(outputs, dim=0)
        output_coords = torch.stack(output_coords, dim=0)
        return outputs, output_coords


class LlavaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = MultiCropImageProcessor
    tokenizer_class = "SiglipTokenizer"

    def __init__(self, image_processor: MultiCropImageProcessor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.search_model = None

    @classmethod
    def from_pretrained(cls, path, trust_remote_code=True, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=trust_remote_code
        )
        image_processor = MultiCropImageProcessor(
            path, trust_remote_code=trust_remote_code
        )
        return LlavaProcessor(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        images: Union[List[Image.Image], None] = None,
        model=None,
        max_crops: int = 0,
        num_tokens=None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if images is not None:
            pixel_values, coords = self.image_processor(images, max_crops)
            pixel_values = pixel_values.to(model.device)
            coords = coords.to(model.device)
            image_outputs = model.vision_model(
                pixel_values, coords, num_tokens=num_tokens
            )
            image_features = model.multi_modal_projector(image_outputs)
        else:
            image_features = None
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        text_inputs["input_ids"] = text_inputs["input_ids"].to(model.device)
        text_inputs["attention_mask"] = text_inputs["attention_mask"].to(model.device)
        return BatchFeature(data={**text_inputs, "image_features": image_features})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
