from io import BytesIO
import random
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from conversation import Conversation
from processing_mc_llava import MultiCropImageProcessor
from PIL import Image
import os
import requests
import pickle


@dataclass
class ImageInfo:
    id: int
    url: str
    landmark_id: int


@dataclass
class LandmarkInfo:
    id: int
    name: str
    latitude: float
    longitude: float
    city: str
    state: str
    country: str
    images: List[ImageInfo]


lat_lon_questions = [
    "Provide me with geographical coordinates of this place.",
    "Can you give me the GPS coordinates of this location?",
    "Could you supply the latitude and longitude for this area?",
    "I need the geographical coordinates of this spot; can you provide them?",
    "What are the latitude and longitude of this place?",
    "Could you share the exact geographical positioning of this location?",
    "Can you give me the coordinates for this specific location?",
    "Could you tell me the GPS location of this site?",
    "I'd appreciate if you could provide the geographical details (latitude and longitude) of this place.",
    "What are the precise coordinates of this area?",
    "Could you help by providing the exact GPS location of this spot?",
    "Please share the latitude and longitude coordinates for this location.",
    "Can you disclose the geographical positioning of this place?",
    "I'm looking for the GPS coordinates of this location; can you assist?",
    "Could you specify the latitude and longitude for this particular spot?",
    "What's the geographical location (in coordinates) of this area?",
    "Could you provide the precise GPS details of this site?",
    "I require the exact geographical coordinates of this location, can you supply them?",
    "Can you tell me the exact latitude and longitude of this place?",
    "Please, could you indicate the GPS coordinates for this specific location?",
    "I need the geographical coordinates for this spot; could you help me with that?",
]

city_questions = [
    "In which city is this place located?",
    "Can you tell me the city where this place is situated?",
    "Could you provide the name of the city this place is located in?",
    "What city can I find this place in?",
    "I'm interested in knowing the city location of this place.",
    "Could you tell me in which city this place is found?",
    "In what city is this place?",
    "Could you share the city where this place is located?",
    "I need to know the city this place is in.",
    "What's the name of the city where this place is located?",
    "Can you pinpoint the city this place is situated in?",
    "Please inform me about the city where this place can be found.",
    "Could you disclose the city this place is located within?",
    "What city is this place a part of?",
    "Can you provide the details of the city this place is located in?",
    "I'm curious about the city this place is located in; can you tell me?",
    "Could you indicate the city where this place is situated?",
    "Can you reveal the city in which this place is found?",
]

country_questions = [
    "In what country is this place located?",
    "Can you tell me the country where this place is situated?",
    "Could you provide the name of the country this place is located in?",
    "What country can I find this place in?",
    "I'm interested in knowing the country location of this place; can you specify?",
    "Please, could you tell me in which country this place is found?",
    "In what country is this place?",
    "Could you share the country where this place is located?",
    "I need to know the country this place is in?",
    "What's the name of the country where this place is located?",
    "I'd appreciate knowing the country where this place is found.",
    "What country is this place a part of?",
    "Can you provide the details of the country this place is located in?",
    "Which country is home to this place? I'm looking for the name.",
    "I'm curious about the country this place is located in; can you tell me?",
    "Please, could you indicate the country where this place is situated?",
]

place_questions = [
    "Where is this place located?",
    "Can you provide the location of this place?",
    "Could you tell me where this place is situated?",
    "What is the specific location of this place?",
    "I'm interested in finding out where this place is located; can you share that information?",
    "Please, could you indicate the whereabouts of this place?",
    "Where exactly can I find this place?",
    "Could you share the precise location of this place?",
    "I need the location of this place; can you provide it?",
    "What's the location of this place?",
    "Can you pinpoint where this place is located?",
    "Please inform me of the location of this place.",
    "Could you disclose the exact location of this place?",
    "Where is this place found?",
    "Can you provide the details of where this place is situated?",
    "What's the geographical location of this place?",
    "Could you tell me the exact location of this place?",
    "I'm curious about the location of this place; can you inform me?",
    "Could you specify the location of this place?",
    "Can you reveal where this place is located?",
]


class LandmarksDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        tokenizer,
        processor: MultiCropImageProcessor,
        crops_limit: int = 8,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
        self.tokenizer = tokenizer
        self.processor = processor
        self.crops_limit = crops_limit
        cache_path = os.path.join(images_dir, "landmarks.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.landmarks = pickle.load(f)
            return
        self.landmarks = []
        landmarks_dataset = load_dataset("visheratin/google_landmarks_places")
        images_dataset = load_dataset("visheratin/google_landmarks_photos")
        images_df = images_dataset["train"].to_pandas()
        for item in landmarks_dataset["train"]:
            landmark = LandmarkInfo(
                id=item["id"],
                name=item["name"],
                latitude=item["lat"],
                longitude=item["lon"],
                city=item["city"],
                state=item["state"],
                country=item["country"],
                images=[],
            )
            image_items = images_df[images_df["landmark_id"] == landmark.id]
            images = image_items.apply(
                lambda x: ImageInfo(
                    id=x["id"], url=x["url"], landmark_id=x["landmark_id"]
                ),
                axis=1,
            )
            landmark.images = images.tolist()
            self.landmarks.append(landmark)
        with open(cache_path, "wb") as f:
            pickle.dump(self.landmarks, f)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        item = self.landmarks[idx]
        question, answer = self.format_item(item)
        question = f"<image>\n{question}"
        conv = Conversation([question, answer])
        _, input_ids, labels = conv.get_prompt(self.tokenizer)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        image = None
        for _ in range(3):
            try:
                image_idx = random.randrange(0, len(item.images))
                image_info = item.images[image_idx]
                image = self.open_image(image_info)
                break
            except:
                continue
        if image is None:
            return self.__getitem__(idx + 1)
        max_crops = random.randint(0, self.crops_limit)
        image_res = self.processor([image], max_crops)
        return (
            input_ids,
            attention_mask,
            labels,
            image_res["pixel_values"],
            image_res["coords"],
        )

    def open_image(self, info: ImageInfo):
        file_path = os.path.join(self.images_dir, f"{info.id}.jpg")
        if os.path.exists(file_path):
            return Image.open(file_path).convert("RGB")
        else:
            request = requests.Request(
                "GET",
                info.url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                },
            )
            prepared_request = request.prepare()
            response = requests.Session().send(prepared_request, timeout=10)
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            image.save(file_path)
            return image

    def format_item(self, item: LandmarkInfo):
        prob = random.random()
        if prob > 0.5:
            question = random.choice(place_questions)
            prob = random.random()
            if prob > 0.5:
                return question, f"({item.latitude}, {item.longitude})"
            else:
                location = ""
                if item.city != "":
                    location += f"{item.city}, "
                if item.state != "":
                    location += f"{item.state}, "
                location += item.country
                return question, location
        else:
            prob = random.random()
            if prob > 0.5:
                question = random.choice(lat_lon_questions)
                return question, f"({item.latitude}, {item.longitude})"
            else:
                prob = random.random()
                if item.city != "" and prob > 0.5:
                    question = random.choice(city_questions)
                    return question, item.city
                else:
                    question = random.choice(country_questions)
                    return question, item.country
