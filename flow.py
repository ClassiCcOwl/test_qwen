from typing import Dict, Union, Optional
from io import BytesIO
from PIL import Image
import logging
import time
import json
import re

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI, UploadFile, Form
import torch


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI()


def extract_json_list(s: str) -> dict:
    logger.info("Extracting JSON from the provided string.")
    pattern = r"```json\s*({[\s\S]*?})\s*```"
    match = re.search(pattern, s)
    if not match:
        logger.warning("No JSON pattern found in the string.")
        return []
    json_str = match.group(1)
    logger.info("JSON pattern successfully extracted.")
    return json.loads(json_str)


class QwenModel:
    def __init__(self, model_address: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_address,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            model_address,
            use_fast=True,
        )
        logger.info("Model and processor loaded successfully.")

    def infer(self, prompt: str, resolution: int, pil_image: Image.Image = None):
        start_time = time.time()
        messages = []

        if pil_image is not None:
            logger.info(f"Original image size: {pil_image.size}")
            
            if resolution:
                original_width, original_height = pil_image.size
                aspect_ratio = original_width / original_height
                if original_width > original_height:
                    new_width = resolution
                    new_height = int(resolution / aspect_ratio)
                else:
                    new_height = resolution
                    new_width = int(resolution * aspect_ratio)
                
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image size: {pil_image.size}")
            
            messages.append({"type": "image", "image": pil_image})
        else:
            pil_image = None

        messages.append({"type": "text", "text": prompt})
        chat_messages = [{"role": "user", "content": messages}]

        logger.info("Preparing inputs for the model.")
        text = self.processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(chat_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logger.info("Inputs prepared. Starting generation.")
        generation_start = time.time()
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generation_end = time.time()
        logger.info(f"Generation completed in {generation_end - generation_start:.4f} seconds.")

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        end_time = time.time()
        logger.info(f"Output generated: {output_text}")
        logger.info(f"Total processing time: {end_time - start_time:.4f} seconds.")

        total_tokens_generated = sum(len(ids) for ids in generated_ids_trimmed)
        average_time_per_token = (generation_end - generation_start) / total_tokens_generated
        logger.info(f"Generated tokens: {total_tokens_generated}. Average time per token: {average_time_per_token:.6f} seconds.")

        if pil_image is not None:
            return output_text, pil_image
        return output_text, None
    

model_instance = QwenModel(model_address="/app/Qwen2.5-VL-7B-Instruct")


@app.post("/detect_human", response_model=Dict[str, Union[str, int]])
async def detect_human_in_image(
    image_file: UploadFile,
    resolution: int = 1024,
    prompt: Optional[str] = Form(None)
) -> Dict[str, Union[str, int]]:
    image_data = await image_file.read()
    pil_image = Image.open(BytesIO(image_data)).convert("RGB")

    # Use default prompt if none provided
    if prompt is None:
        prompt = (
            "Is there a human present in this image? Also return number of humans. "
            "Answer in JSON format like this:\n```json\n"
            "{\"human\": \"yes/no\", \"number\": integer}\n```"
        )

    output_text, _ = model_instance.infer(
        prompt=prompt,
        resolution=resolution,
        pil_image=pil_image,
    )
    return extract_json_list(output_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)