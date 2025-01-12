import os
import replicate
import pandas as pd
from pathlib import Path
from glob import glob
import json
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_image_to_imgbb(image_path, api_key):
    try:
        with open(image_path, "rb") as file:
            base64_image = base64.b64encode(file.read()).decode('utf-8')
        
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
            "image": base64_image,
            "expiration": 600
        }
        
        response = requests.post(url, payload)
        response.raise_for_status()
        return response.json()["data"]["url"]
    except Exception as e:
        print(f"Error uploading image {image_path}: {str(e)}")
        raise

def load_eval_dir():
    with open("artifacts/eval_dir.txt", "r") as f:
        return Path(f.read().strip())

def get_image_files(eval_dir):
    models = [d for d in eval_dir.iterdir() if d.is_dir()]
    image_files = {}
    for model_dir in models:
        image_files[model_dir.name] = sorted(glob(str(model_dir / "*.png"))) + sorted(glob(str(model_dir / "*.jpg")))
    return image_files

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_replicate_eval(prompts_and_images):
    try:
        return replicate.run(
            "andreasjansson/flash-eval:ef9f9879404379fd05006e888a6bddcab189914c1045b77cbabc565735164ce9",
            input={
                "models": "ImageReward,Aesthetic,CLIP,BLIP,PickScore",
                "image_separator": ",",
                "prompts_and_images": "\n".join(prompts_and_images),
                "prompt_images_separator": ":"
            }
        )
    except Exception as e:
        print(f"Error running evaluation: {str(e)}")
        raise

def evaluate_batch(image_batch, start_idx, imgbb_api_key):
    prompts_and_images = []
    batch_files = []
    url_to_file = {}
    failed_images = []
    
    print(f"Evaluating batch of {len(image_batch)} image sets...")
    
    for idx, images in enumerate(image_batch):
        images_for_prompt = []
        for img_path in images:
            print(f"Uploading image {img_path}...")
            try:
                img_url = upload_image_to_imgbb(img_path, imgbb_api_key)
                images_for_prompt.append(img_url)
                batch_files.append(img_path)
                url_to_file[img_url] = img_path
            except Exception as e:
                print(f"Failed to upload {img_path} after retries. Skipping... Error: {str(e)}")
                failed_images.append(img_path)
                continue
        
        if images_for_prompt:
            prompt = f"image_{start_idx + idx}"
            prompts_and_images.append(f"{prompt}: {','.join(images_for_prompt)}")
    
    if not prompts_and_images:
        print("No images successfully uploaded in this batch")
        return None, url_to_file, failed_images
    
    try:
        output = run_replicate_eval(prompts_and_images)
        return output, url_to_file, failed_images
    except Exception as e:
        print(f"Batch evaluation failed after retries: {str(e)}")
        return None, url_to_file, failed_images

def evaluate_images(image_files):
    imgbb_api_key = os.getenv("IMGBB_API_KEY")
    if not imgbb_api_key:
        raise ValueError("IMGBB_API_KEY environment variable is required")
    
    batch_size = 10
    all_batches = []
    all_failed_images = []
    
    max_images = max(len(files) for files in image_files.values())
    for idx in range(0, max_images, batch_size):
        batch = []
        for i in range(batch_size):
            if idx + i >= max_images:
                break
            images_for_prompt = []
            for model in image_files.keys():
                if idx + i < len(image_files[model]):
                    images_for_prompt.append(image_files[model][idx + i])
            if images_for_prompt:
                batch.append(images_for_prompt)
        if batch:
            all_batches.append(batch)
    
    all_results = []
    model_scores = {model: {metric: [] for metric in ["BLIP", "CLIP", "Aesthetic", "PickScore", "ImageReward"]} 
                   for model in image_files.keys()}
    
    for batch_idx, batch in enumerate(all_batches):
        print(f"\nProcessing batch {batch_idx + 1} of {len(all_batches)}...")
        output, url_to_file, failed_images = evaluate_batch(batch, batch_idx * batch_size, imgbb_api_key)
        all_failed_images.extend(failed_images)
        
        if not output:
            print(f"Skipping batch {batch_idx + 1} due to evaluation failure")
            continue
        
        for result in output:
            prompt_name = result['prompt']
            scores = result['scores']
            
            for image_url, metrics in scores.items():
                orig_file = url_to_file.get(image_url)
                if orig_file:
                    model = orig_file.split(os.sep)[-2]
                    filename = os.path.basename(orig_file)
                    
                    result = {
                        'model': model,
                        'filename': filename,
                        **metrics
                    }
                    all_results.append(result)
                    print(f"Processed {model}/{filename}")
                    
                    for metric, score in metrics.items():
                        model_scores[model][metric].append(score)
        
        if batch_idx < len(all_batches) - 1:
            time.sleep(2)
    
    # Save failed images list
    if all_failed_images:
        failed_path = Path(eval_dir) / "failed_images.txt"
        with open(failed_path, 'w') as f:
            for img in all_failed_images:
                f.write(f"{img}\n")
        print(f"\nList of failed images saved to: {failed_path}")
    
    print("\nAverage scores by model:")
    print("-" * 50)
    for model in model_scores:
        print(f"\nModel: {model}")
        for metric in model_scores[model]:
            scores = model_scores[model][metric]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"{metric}: {avg_score:.4f}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        results_path = Path(eval_dir) / "results.csv"
        df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to: {results_path}")
    else:
        print("\nNo results were generated due to evaluation failures")

if __name__ == "__main__":
    try:
        eval_dir = load_eval_dir()
        print(f"Evaluating images in: {eval_dir}")
        
        image_files = get_image_files(eval_dir)
        if not image_files:
            print("No image files found in the evaluation directory")
        else:
            evaluate_images(image_files)
    except Exception as e:
        print(f"Script failed with error: {str(e)}")