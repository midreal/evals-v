import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any
import base64
from io import BytesIO
from PIL import Image
import time
from dotenv import load_dotenv

class BaseModel(ABC):
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('SEGMIND_API_KEY')
        if not self.api_key:
            raise ValueError("SEGMIND_API_KEY not found in environment variables")

    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, Any]:
        pass

    def save_image(self, image_data: bytes, output_path: str):
        img = Image.open(BytesIO(image_data))
        img.save(output_path)
        return output_path

class FluxModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.url = "https://api.segmind.com/v1/flux-dev"
        self.model_name = "flux"

    def generate(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "negative_prompt": "nsfw, nude, naked, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 20,
            "guidance_scale": 7,
            "width": 512,
            "height": 512,
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            return {
                "success": True,
                "model": self.model_name,
                "image_data": response.content,
                "prompt": prompt
            }
        else:
            return {
                "success": False,
                "model": self.model_name,
                "error": f"Error: {response.status_code} - {response.text}",
                "prompt": prompt
            }

class PlayboyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.url = "https://api.segmind.com/v1/2a2d9f95-b0c4-472f-854d-312a92114793-playboy"
        self.model_name = "playboy"

    def generate(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "negative_prompt": "nsfw, nude, naked, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 20,
            "guidance_scale": 7,
            "width": 512,
            "height": 512,
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            return {
                "success": True,
                "model": self.model_name,
                "image_data": response.content,
                "prompt": prompt
            }
        else:
            return {
                "success": False,
                "model": self.model_name,
                "error": f"Error: {response.status_code} - {response.text}",
                "prompt": prompt
            }
