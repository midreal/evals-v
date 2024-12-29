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

# Load environment variables
load_dotenv()

def upload_image_to_imgbb(image_path, api_key):
    with open(image_path, "rb") as file:
        # Convert image to base64
        base64_image = base64.b64encode(file.read()).decode('utf-8')
    
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": api_key,
        "image": base64_image,
        "expiration": 60  # Delete after 1 minute
    }
    
    response = requests.post(url, payload)
    if response.status_code == 200:
        return response.json()["data"]["url"]
    else:
        raise Exception(f"Failed to upload image: {response.text}")

def load_eval_dir():
    with open("artifacts/eval_dir.txt", "r") as f:
        return Path(f.read().strip())

def get_image_files(eval_dir):
    models = [d for d in eval_dir.iterdir() if d.is_dir() and d.name in ['flux', 'playboy']]
    image_files = {}
    for model_dir in models:
        image_files[model_dir.name] = sorted(glob(str(model_dir / "*.png")))
    return image_files

def evaluate_batch(image_batch, start_idx, imgbb_api_key):
    prompts_and_images = []
    batch_files = []
    url_to_file = {}  # Map URLs to original files
    
    print(f"Evaluating batch of {len(image_batch)} image sets...")
    
    for idx, images in enumerate(image_batch):
        images_for_prompt = []
        for img_path in images:
            # Upload image and get URL
            img_url = upload_image_to_imgbb(img_path, imgbb_api_key)
            images_for_prompt.append(img_url)
            batch_files.append(img_path)
            url_to_file[img_url] = img_path
        
        if images_for_prompt:
            # Use the index as a simple prompt
            prompt = f"image_{start_idx + idx}"
            prompts_and_images.append(f"{prompt}: {','.join(images_for_prompt)}")
    
    # Run evaluation
    print(f"Sending evaluation request for {len(prompts_and_images)} image sets...")
    
    output = replicate.run(
        "andreasjansson/flash-eval:ef9f9879404379fd05006e888a6bddcab189914c1045b77cbabc565735164ce9",
        input={
            "models": "ImageReward,Aesthetic,CLIP,BLIP,PickScore",
            "image_separator": ",",
            "prompts_and_images": "\n".join(prompts_and_images),
            "prompt_images_separator": ":"
        }
    )
    
    return output, url_to_file

def evaluate_images(image_files):
    # Check for ImgBB API key
    imgbb_api_key = os.getenv("IMGBB_API_KEY")
    if not imgbb_api_key:
        raise ValueError("IMGBB_API_KEY environment variable is required")
    
    # Prepare batches
    batch_size = 5  # Process 5 prompts at a time
    all_batches = []
    
    # Group corresponding images from different models
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
    
    # Process all batches
    all_results = []
    model_scores = {model: {metric: [] for metric in ["BLIP", "CLIP", "Aesthetic", "PickScore", "ImageReward"]} 
                   for model in image_files.keys()}
    
    for batch_idx, batch in enumerate(all_batches):
        output, url_to_file = evaluate_batch(batch, batch_idx * batch_size, imgbb_api_key)
        
        if not output:
            print(f"Warning: No output received for batch {batch_idx + 1}")
            continue
        
        # Process results
        print(f"Processing results for batch {batch_idx + 1}...")
        
        for result in output:
            prompt_name = result['prompt']
            scores = result['scores']
            
            for image_url, metrics in scores.items():
                # Get original file path and model name
                orig_file = url_to_file.get(image_url)
                if orig_file:
                    model = orig_file.split(os.sep)[-2]  # Get model name from path
                    filename = os.path.basename(orig_file)
                    
                    # Add to results
                    result = {
                        'model': model,
                        'filename': filename,
                        **metrics
                    }
                    all_results.append(result)
                    print(f"\nScores for {model}/{filename}:")
                    for metric, score in metrics.items():
                        print(f"{metric}: {score:.4f}")
                    
                    # Add to model scores
                    for metric, score in metrics.items():
                        model_scores[model][metric].append(score)
        
        # Add delay between batches
        if batch_idx < len(all_batches) - 1:
            time.sleep(2)
    
    # Calculate and print average scores for each model
    print("\nAverage scores by model:")
    print("-" * 50)
    for model in model_scores:
        print(f"\nModel: {model}")
        for metric in model_scores[model]:
            scores = model_scores[model][metric]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"{metric}: {avg_score:.4f}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    results_path = Path(eval_dir) / "results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    eval_dir = load_eval_dir()
    print(f"Evaluating images in: {eval_dir}")
    
    image_files = get_image_files(eval_dir)
    if not image_files:
        print("No image files found in the evaluation directory")
    else:
        evaluate_images(image_files)
