#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time

import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType
from core.blora_helper import BLoRAConfig, TrainingType, analyze_training_requirements


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path


def count_images_in_directory(directory_path: str) -> int:
    """Count the number of image files in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                # Check if file has an image extension
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count


def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None


def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    """Get configuration overrides based on model name."""
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None


def apply_reg_ratio_to_lr(value, reg_ratio: float):
    """
    Scale learning-rate values (single value or list/tuple of values) by reg_ratio.
    Leaves non-numeric values unchanged.
    """
    if value is None:
        return None

    def _scale(v):
        if isinstance(v, (int, float)):
            return v * reg_ratio
        if isinstance(v, str):
            try:
                return float(v) * reg_ratio
            except ValueError:
                return v
        return v

    if isinstance(value, (list, tuple)):
        return [_scale(v) for v in value]

    return _scale(value)


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
OOM_ERROR_ALT = "OutOfMemoryError"


def get_error_type(log_path: str):
    """Check if log file contains OOM error"""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, "r") as f:
        text = f.read()
    
    if OOM_ERROR in text or OOM_ERROR_ALT in text:
        return OOM_ERROR
    
    return None


def reduce_batch_size_in_config(config: dict, reduction_factor: int = 2) -> bool:
    """
    Reduce batch size in config dictionary.
    Returns True if reduction was successful, False if batch size is already at minimum.
    """
    current_batch_size = config.get("train_batch_size")
    current_max_data_loader_n_workers = config.get("max_data_loader_n_workers")

    if current_batch_size > 1:
        new_batch_size = max(1, current_batch_size // reduction_factor)
        config["train_batch_size"] = new_batch_size
        print(f"Reducing batch size from {current_batch_size} to {new_batch_size}", flush=True)
        return True
    elif current_max_data_loader_n_workers > 1:
        new_max_data_loader_n_workers = max(1, current_max_data_loader_n_workers // reduction_factor)
        config["max_data_loader_n_workers"] = new_max_data_loader_n_workers
        print(f"Reducing gradient accumulation steps from {current_max_data_loader_n_workers} to {new_gradient_accumulation_steps}", flush=True)
        return True
    else:
        print(f"Batch size is already 1, cannot reduce further", flush=True)
        return False


def reduce_gradient_accumulation_in_config(config: dict) -> bool:
    """
    Reduce gradient accumulation steps in config dictionary.
    Returns True if reduction was successful, False if already at minimum.
    """
    current_grad_accum = config.get("gradient_accumulation_steps", 1)
    
    if current_grad_accum > 1:
        new_grad_accum = max(1, current_grad_accum // 2)
        config["gradient_accumulation_steps"] = new_grad_accum
        print(f"Reducing gradient accumulation steps from {current_grad_accum} to {new_grad_accum}", flush=True)
        return True
    else:
        print(f"Gradient accumulation is already 1, cannot reduce further", flush=True)
        return False

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, reg_ratio=1.0):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    normalized_model_type = (model_type or "").lower()
    
    # Load and apply LRS configuration
    lrs_config = load_lrs_config(model_type, is_style)
    if lrs_config:
        model_hash = hash_model(model_name)
        lrs_settings = get_config_for_model(lrs_config, model_hash)

        if lrs_settings:
            base_unet_lr = lrs_settings.get('unet_lr')
            base_text_encoder_lr = lrs_settings.get('text_encoder_lr')

            final_unet_lr = base_unet_lr * reg_ratio if base_unet_lr else None
            final_text_encoder_lr = apply_reg_ratio_to_lr(base_text_encoder_lr, reg_ratio)

            print(f"Applying LRS configuration for model '{model_name}' (hash: {model_hash}):", flush=True)
            print(f"  - Base unet_lr: {base_unet_lr} × reg_ratio: {reg_ratio} = {final_unet_lr}", flush=True)
            print(f"  - Base text_encoder_lr: {base_text_encoder_lr} × reg_ratio: {reg_ratio} = {final_text_encoder_lr}", flush=True)
            print(f"  - train_batch_size: {lrs_settings.get('train_batch_size')}", flush=True)
            print(f"  - gradient_accumulation_steps: {lrs_settings.get('gradient_accumulation_steps')}", flush=True)
            print(f"  - min_snr_gamma: {lrs_settings.get('min_snr_gamma')}", flush=True)
            print(f"  - lr_warmup_steps: {lrs_settings.get('lr_warmup_steps')}", flush=True)
            print(f"  - max_grad_norm: {lrs_settings.get('max_grad_norm')}", flush=True)
            print(f"  - max_train_epochs: {lrs_settings.get('max_train_epochs')}", flush=True)
            print(f"  - network_alpha: {lrs_settings.get('network_alpha')}", flush=True)
            print(f"  - network_dim: {lrs_settings.get('network_dim')}", flush=True)
            print(f"  - network_args: {lrs_settings.get('network_args')}", flush=True)
            print(f"  - max_train_steps: {lrs_settings.get('max_train_steps')}", flush=True)


            if final_unet_lr is not None:
                config['unet_lr'] = final_unet_lr
            if final_text_encoder_lr is not None:
                config['text_encoder_lr'] = final_text_encoder_lr
            if lrs_settings.get('train_batch_size') is not None:
                config['train_batch_size'] = lrs_settings.get('train_batch_size')
            if lrs_settings.get('gradient_accumulation_steps') is not None:
                config['gradient_accumulation_steps'] = lrs_settings.get('gradient_accumulation_steps')
            if lrs_settings.get('min_snr_gamma') is not None:
                config['min_snr_gamma'] = lrs_settings.get('min_snr_gamma')
            if lrs_settings.get('lr_warmup_steps') is not None:
                config['lr_warmup_steps'] = lrs_settings.get('lr_warmup_steps')
            if lrs_settings.get('max_grad_norm') is not None:
                config['max_grad_norm'] = lrs_settings.get('max_grad_norm')
            if lrs_settings.get('max_train_epochs') is not None:
                config['max_train_epochs'] = lrs_settings.get('max_train_epochs')
            if lrs_settings.get('network_alpha') is not None:
                config['network_alpha'] = lrs_settings.get('network_alpha')
            if lrs_settings.get('network_dim') is not None:
                config['network_dim'] = lrs_settings.get('network_dim')
            if lrs_settings.get('network_args') is not None:
                config['network_args'] = lrs_settings.get('network_args')
            if lrs_settings.get('max_train_steps') is not None:
                config['max_train_steps'] = lrs_settings.get('max_train_steps')

            for optional_key in [
                "train_batch_size",
                "max_data_loader_n_workers",
                "optimizer_args",
                "min_snr_gamma",
                "prior_loss_weight",
                "max_grad_norm",
                "network_alpha",
                "network_dim",
                "network_args",
            ]:
                if optional_key in lrs_settings:
                    config[optional_key] = lrs_settings[optional_key]
        else:
            print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
    else:
        print("Warning: Could not load LRS configuration, using default values", flush=True)

    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path, log_path):
    """
    Run training subprocess and log output.
    Returns True if training completed successfully, False otherwise.
    """
    print(f"Starting training with config: {config_path}", flush=True)

    if model_type == "sdxl":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-script/{model_type}_train_network.py",
            "--config_file", config_path
        ]
    elif model_type == "flux":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-scripts/{model_type}_train_network.py",
            "--config_file", config_path
        ]

    try:
        print("Starting training subprocess...\n", flush=True)
        
        with open(log_path, "w") as log_file:
            log_file.write(f"Training command: {' '.join(training_command)}\n")
            log_file.write("=" * 80 + "\n")
            log_file.flush()
            
            process = subprocess.Popen(
                training_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()
            
            log_file.write(f"\nProcess completed with return code: {return_code}\n")
            log_file.flush()
            
            if return_code != 0:
                print("Training subprocess failed!", flush=True)
                print(f"Exit Code: {return_code}", flush=True)
                return False

        print("Training subprocess completed successfully.", flush=True)
        return True

    except Exception as e:
        print(f"Training subprocess encountered an error: {e}", flush=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"\nException occurred: {str(e)}\n")
        return False


def check_training_success(output_dir: str) -> bool:
    """
    Check if training completed successfully by verifying output directory.
    Returns True if training artifacts are present and best checkpoint info exists.
    """
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}", flush=True)
        return False
    
    # Check if there are any .safetensors files in the output directory
    safetensors_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
    
    if len(safetensors_files) == 0:
        print("No .safetensors files found in output directory", flush=True)
        return False
    
    # Check for last.safetensors (the final best checkpoint)
    last_checkpoint = os.path.join(output_dir, "last.safetensors")
    if os.path.exists(last_checkpoint):
        print(f"✅ Found final checkpoint: last.safetensors", flush=True)
        
        # Check for best checkpoint metadata
        metadata_file = os.path.join(output_dir, "best_checkpoint_info.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    info = json.load(f)
                    print(f"✅ Best checkpoint info:", flush=True)
                    print(f"   - Epoch: {info.get('best_epoch', 'N/A')}", flush=True)
                    print(f"   - Loss: {info.get('best_loss', 'N/A')}", flush=True)
                    print(f"   - Total epochs: {info.get('total_epochs', 'N/A')}", flush=True)
            except Exception as e:
                print(f"Warning: Could not read best checkpoint metadata: {e}", flush=True)
        
        return True
    else:
        print(f"Warning: last.safetensors not found. Available files: {safetensors_files}", flush=True)
        # Still consider it a success if there are other checkpoint files
        return True
    
def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--retries", type=int, default=5, help="Number of retries on OOM error")
    parser.add_argument("--reg-ratio", type=float, help="Reg ratio to use for training", default=0.917233)
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)
    output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    reg_ratio = args.reg_ratio
    
    # Create logs directory
    logs_dir = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"train_{args.task_id}.log")

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Create initial config file (will count images and apply LRS config)
    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        reg_ratio,
    )

    # Load the config for potential modifications
    with open(config_path, "r") as f:
        config = toml.load(f)

    # Training loop with retry logic for OOM errors
    train_success = False
    for attempt in range(args.retries):
        print(f"\n{'='*80}", flush=True)
        print(f"Training attempt {attempt + 1}/{args.retries} for task {args.task_id}", flush=True)
        print(f"{'='*80}\n", flush=True)

        # If this is a retry (not the first attempt), handle OOM error
        if attempt > 0:
            error_type = get_error_type(log_path)
            
            if error_type == OOM_ERROR:
                print(f"Detected OOM error on attempt {attempt}. Applying memory optimizations...", flush=True)
                
                optimization_applied = False
                
                # Strategy 1: Reduce batch size (works for both SDXL and Flux)
                if reduce_batch_size_in_config(config):
                    optimization_applied = True
                # Strategy 2: Reduce gradient accumulation (for Flux mainly)
                elif args.model_type == "flux" and reduce_gradient_accumulation_in_config(config):
                    optimization_applied = True
                
                if not optimization_applied:
                    print("All memory optimization strategies exhausted. Cannot proceed.", flush=True)
                    break
                
                # Save updated config and retry training
                save_config_toml(config, config_path)
                print(f"Updated config saved to {config_path}", flush=True)
                print(f"Will retry training with optimized settings...", flush=True)
            else:
                print(f"Training failed on attempt {attempt} but no OOM error detected. Retrying...", flush=True)
        
        # Clear/reset log file
        with open(log_path, "w") as f:
            f.write(f"STARTING TRAINING - Attempt {attempt + 1}/{args.retries}\n")
            f.write(f"Config: {config_path}\n")
            f.write("=" * 80 + "\n")

        # Run training
        success = run_training(args.model_type, config_path, log_path)
        
        # Wait a bit before checking results
        time.sleep(3)
        
        # Check if training was successful
        if success and check_training_success(output_dir):
            print(f"\nTraining completed successfully for task {args.task_id}!", flush=True)
            train_success = True
            break
        else:
            print(f"\nTraining attempt {attempt + 1} failed or produced no output.", flush=True)
            if attempt < args.retries - 1:
                print(f"Retrying...\n", flush=True)
                time.sleep(2)
    
    if not train_success:
        print(f"\n{'='*80}", flush=True)
        print(f"Training failed for task {args.task_id} after {args.retries} attempts", flush=True)
        print(f"{'='*80}\n", flush=True)
        raise RuntimeError(f"Training failed after {args.retries} attempts")
    
    print(f"\n{'='*80}", flush=True)
    print(f"Training pipeline completed successfully for task {args.task_id}", flush=True)
    print(f"Output saved to: {output_dir}", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())