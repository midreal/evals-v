import os
import pandas as pd
from models import FluxModel, PlayboyModel
from datetime import datetime
import time
from pathlib import Path
import concurrent.futures
from typing import Dict, Any
import argparse

def create_output_dirs():
    # Create outputs directory if it doesn't exist
    output_base = Path("outputs")
    output_base.mkdir(exist_ok=True)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / timestamp
    run_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each model
    (run_dir / "flux").mkdir(exist_ok=True)
    (run_dir / "playboy").mkdir(exist_ok=True)
    
    return run_dir

def generate_html(results, run_dir):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Results</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px;
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
            }
            .comparison { 
                margin-bottom: 20px;
                border: 1px solid #ddd;
                padding: 15px;
                background: #f9f9f9;
                border-radius: 5px;
            }
            .images { 
                display: flex; 
                gap: 20px; 
                margin-top: 10px;
                justify-content: space-between;
            }
            .model-result { 
                flex: 1;
                min-width: 0;  /* Allow shrinking below content size */
            }
            img { 
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
                border-radius: 5px;
            }
            .prompt { 
                font-size: 0.9em;
                margin-bottom: 10px;
                color: #444;
                font-weight: 500;
            }
            .error { 
                color: #d32f2f;
                padding: 10px;
                background: #ffebee;
                border-radius: 4px;
            }
            h1 {
                color: #333;
                margin-bottom: 30px;
                text-align: center;
            }
            h3 {
                color: #1976d2;
                margin: 10px 0;
                font-size: 1em;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(700px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            @media (max-width: 1400px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <h1>Model Comparison Results</h1>
        <div class="grid">
    """
    
    for result in results:
        html_content += f"""
        <div class="comparison">
            <div class="prompt">{result['prompt']}</div>
            <div class="images">
        """
        
        for model_name in ['flux', 'playboy']:
            html_content += f"""
                <div class="model-result">
                    <h3>{model_name.capitalize()} Model</h3>
            """
            
            if result[model_name]['success']:
                image_path = f"{model_name}/{result[model_name]['image_path'].name}"
                html_content += f'<img src="{image_path}" alt="{model_name} result">'
            else:
                html_content += f'<div class="error">{result[model_name]["error"]}</div>'
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(run_dir / "comparison.html", "w") as f:
        f.write(html_content)

def generate_image(prompt: str, model_name: str, model: Any, run_dir: Path, index: int) -> Dict[str, Any]:
    print(f"> Generating image for prompt: '{prompt}' using {model_name} model...")
    
    try:
        generation_result = model.generate(prompt)
        
        if generation_result['success']:
            image_path = run_dir / model_name / f"{index:03d}.png"
            model.save_image(generation_result['image_data'], image_path)
            generation_result['image_path'] = image_path
        
        return {
            model_name: generation_result,
            'prompt': prompt
        }
        
    except Exception as e:
        return {
            model_name: {
                'success': False,
                'error': str(e),
                'model': model_name,
                'prompt': prompt
            },
            'prompt': prompt
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate models with prompts from a CSV file.")
    parser.add_argument('--num', type=int, default=None, help='Maximum number of prompts to process')
    args = parser.parse_args()

    # Load prompts from CSV
    df = pd.read_csv("artifacts/example.csv", skip_blank_lines=True)
    
    print(df)

    # Initialize models
    models = {
        'flux': FluxModel(),
        'playboy': PlayboyModel()
    }
    
    # Create output directories
    run_dir = create_output_dirs()
    
    # Save the run directory to eval_dir.txt
    with open("artifacts/eval_dir.txt", "w") as f:
        f.write(str(run_dir))
    
    # Store results
    results = []
    combined_results = {}
    failed_prompts = []
    
    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        # Process each prompt
        for index, row in df.iterrows():
            if args.num is not None and index >= args.num:
                break

            prompt = row['prompt'].strip('"')  # Remove quotes from the prompt
            if pd.isna(prompt):  # Skip empty prompts
                continue

            print(f"Processing prompt: '{prompt}'")
            
            # Submit tasks for both models
            prompt_futures = []
            for model_name, model in models.items():
                future = executor.submit(generate_image, prompt, model_name, model, run_dir, index)
                prompt_futures.append((model_name, future))
            futures.append((prompt, index, prompt_futures))
        
        # Collect results as they complete
        for prompt, index, prompt_futures in futures:
            # Wait for all models to complete for this prompt
            prompt_results = {}
            prompt_failed = False
            
            for model_name, future in prompt_futures:
                try:
                    result = future.result(timeout=60)  # Add timeout to prevent hanging
                    if not result[model_name].get('success', False):
                        print(f"Error with {model_name} model for prompt: '{prompt}'")
                        print(f"Error message: {result[model_name].get('error', 'Unknown error')}")
                        prompt_failed = True
                        break
                    prompt_results.update(result)
                except Exception as e:
                    print(f"Exception with {model_name} model for prompt: '{prompt}'")
                    print(f"Error: {str(e)}")
                    prompt_failed = True
                    break
            
            if prompt_failed:
                failed_prompts.append((index, prompt))
                continue
                
            # Only add successful results
            combined_results[prompt] = prompt_results
            
            # Add delay between prompts to avoid rate limiting
            time.sleep(1)
    
    # Convert combined results to list
    results = list(combined_results.values())
    
    # Generate comparison HTML
    if results:
        generate_html(results, run_dir)
        print(f"\nEvaluation complete! Results saved in {run_dir}")
        print(f"Open {run_dir}/comparison.html to view the results")
    else:
        print("\nNo successful results to generate HTML")
    
    # Report failed prompts
    if failed_prompts:
        print("\nFailed prompts:")
        for index, prompt in failed_prompts:
            print(f"Index {index}: {prompt}")

if __name__ == "__main__":
    main()
