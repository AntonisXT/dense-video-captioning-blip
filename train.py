"""
Fine-tuning FLAN-T5 for Video Captioning with MSR-VTT Dataset

This script implements fine-tuning of the FLAN-T5 model for video summarization,
using the MSR-VTT dataset. It supports Parameter-Efficient Fine-Tuning (PEFT)
techniques like LoRA for efficient training.
"""

import os
import argparse
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
import evaluate
import nltk
from nltk.tokenize import sent_tokenize

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetuning/logs/training.log")
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Random seeds set to {seed}")


def load_scene_captions_from_results(results_dir):
    """
    Load scene captions from result JSON files.
    
    Args:
        results_dir: Directory containing result folders for each video
        
    Returns:
        Dictionary mapping video_id to list of scene captions
    """
    scene_captions_dict = {}
    
    # Walk through the results directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_captions.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    video_id = data.get('video_id')
                    if not video_id:
                        continue
                        
                    # Extract just the caption text from each scene
                    captions = [scene['caption'] for scene in data.get('scene_captions', [])]
                    
                    if captions:
                        scene_captions_dict[video_id] = captions
                        
                except Exception as e:
                    logger.error(f"Error loading captions from {file_path}: {e}")
    
    logger.info(f"Loaded scene captions for {len(scene_captions_dict)} videos")
    return scene_captions_dict


def load_msrvtt(train_val_path, test_path=None, blip_scene_captions=None, max_train_videos=None, seed=42):
    """
    Load the MSR-VTT dataset and prepare data for fine-tuning.
    
    Args:
        train_val_path: Path to the MSR-VTT train/val dataset JSON file
        test_path: Path to the MSR-VTT test dataset JSON file (optional)
        blip_scene_captions: Dictionary mapping video_id to list of scene captions
        max_train_videos: Maximum number of training videos to use (for subset training)
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    logger.info(f"Loading MSR-VTT train/val dataset from {train_val_path}")
    
    # Load train/val JSON data
    try:
        with open(train_val_path, 'r', encoding='utf-8') as f:
            train_val_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading train/val dataset: {e}")
        raise
    
    # Load test JSON data if provided
    test_data = None
    if test_path and os.path.exists(test_path):
        logger.info(f"Loading MSR-VTT test dataset from {test_path}")
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            logger.warning("Proceeding without test data")
    
    # Process train/val data
    train_val_videos = train_val_data['videos']
    train_val_annotations = train_val_data['sentences']
    
    # Organize annotations by video_id
    video_annotations = {}
    for ann in train_val_annotations:
        video_id = ann['video_id']
        if video_id not in video_annotations:
            video_annotations[video_id] = []
        video_annotations[video_id].append(ann['caption'])
    
    # Process test data if available
    if test_data:
        test_videos_data = test_data.get('videos', [])
        test_annotations = test_data.get('sentences', [])
        
        # Organize test annotations by video_id
        for ann in test_annotations:
            video_id = ann['video_id']
            if video_id not in video_annotations:
                video_annotations[video_id] = []
            video_annotations[video_id].append(ann['caption'])
        
        # Add test videos to the list
        train_val_videos.extend(test_videos_data)
    
    # Find all videos that have scene captions
    videos_with_captions = []
    for video in train_val_videos:
        video_id = video['video_id']
        if video_id in blip_scene_captions and len(blip_scene_captions[video_id]) > 0:
            videos_with_captions.append(video_id)
    
    logger.info(f"Found {len(videos_with_captions)} videos with scene captions")
    
    if len(videos_with_captions) == 0:
        raise ValueError("No videos with scene captions found. Cannot proceed with training.")
    
    # Split videos with captions into train/val/test
    random.seed(seed)
    random.shuffle(videos_with_captions)
    
    # Calculate split sizes
    train_size = int(0.7 * len(videos_with_captions))
    val_size = int(0.15 * len(videos_with_captions))
    # test_size will be the remaining videos
    
    # Apply max_train_videos limit if specified
    if max_train_videos and train_size > max_train_videos:
        train_size = max_train_videos
    
    train_videos = videos_with_captions[:train_size]
    val_videos = videos_with_captions[train_size:train_size+val_size]
    test_videos = videos_with_captions[train_size+val_size:]
    
    logger.info(f"Split {len(videos_with_captions)} videos with captions into:")
    logger.info(f"  Train: {len(train_videos)} videos")
    logger.info(f"  Validation: {len(val_videos)} videos")
    logger.info(f"  Test: {len(test_videos)} videos")
    
    # Create dataset rows
    dataset_rows = []
    
    # Function to process videos and create dataset entries
    def process_videos(videos, split_name):
        rows = []
        
        for video_id in tqdm(videos, desc=f"Processing {split_name} videos"):
            if video_id in video_annotations and video_id in blip_scene_captions:
                # Get ground truth captions
                gt_captions = video_annotations[video_id]
                
                # Get scene captions
                scene_captions = blip_scene_captions[video_id]
                
                # For each ground truth caption, create a training example
                for gt_caption in gt_captions:
                    input_text = f"Summarize the video in one short and natural sentence based on these scene descriptions:\n{' '.join(scene_captions)}"
                    
                    # Add to dataset
                    row = {
                        'video_id': video_id,
                        'input_text': input_text,
                        'target_text': gt_caption,
                        'scene_captions': scene_captions,
                        'split': split_name
                    }
                    
                    rows.append(row)
        return rows
    
    # Process each split
    train_rows = process_videos(train_videos, 'train')
    val_rows = process_videos(val_videos, 'validation')
    test_rows = process_videos(test_videos, 'test')
    
    # Combine all rows
    dataset_rows = train_rows + val_rows + test_rows
    
    logger.info(f"Created {len(dataset_rows)} dataset entries")
    logger.info(f"  Train: {len(train_rows)} entries")
    logger.info(f"  Validation: {len(val_rows)} entries")
    logger.info(f"  Test: {len(test_rows)} entries")
    
    # Convert to DataFrame and then to Dataset
    df = pd.DataFrame(dataset_rows)
    
    # Split by the 'split' column
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict



def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """Preprocess dataset examples for fine-tuning."""
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        padding="max_length", 
        truncation=True
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        padding="max_length", 
        truncation=True
    )
    
    # Replace padding token id with -100 so it's ignored in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Calculate evaluation metrics for model predictions."""
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    preds, labels = eval_preds
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 with pad token id
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Calculate ROUGE scores
    rouge_results = rouge_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    # Calculate METEOR score
    meteor_result = meteor_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels
    )
    
    # Combine results
    result = {
        **{k: round(v * 100, 4) for k, v in rouge_results.items()},
        "meteor": round(meteor_result["meteor"] * 100, 4)
    }
    
    return result


def main():
    """Main function for fine-tuning FLAN-T5 for video captioning."""
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 for video captioning")
    parser.add_argument("--train_val_path", type=str, default="data/captions/train_val_videodatainfo.json",
                        help="Path to the MSR-VTT train/val dataset JSON file")
    parser.add_argument("--test_path", type=str, default="data/captions/test_videodatainfo.json",
                        help="Path to the MSR-VTT test dataset JSON file")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing result JSON files with scene captions")
    parser.add_argument("--processed_data_path", type=str, default="finetuning/data/processed/msrvtt_dataset",
                        help="Path to save/load the processed dataset")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="finetuning/models/flan-t5-msrvtt",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--max_train_videos", type=int, default=None,
                        help="Maximum number of training videos to use (for subset training)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for training")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=64,
                        help="Maximum target sequence length")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Whether to use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Whether to use mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging steps during training")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation steps during training")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to save")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Force reprocessing of the dataset even if it exists")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Create necessary directories
    os.makedirs(os.path.dirname(args.processed_data_path), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("finetuning/results", exist_ok=True)
    os.makedirs("finetuning/logs", exist_ok=True)
    
    # Load scene captions from results if provided
    blip_scene_captions = {}
    if args.results_dir and os.path.exists(args.results_dir):
        logger.info(f"Loading scene captions from results directory: {args.results_dir}")
        blip_scene_captions = load_scene_captions_from_results(args.results_dir)
    
    # Prepare dataset
    if os.path.exists(args.processed_data_path) and not args.force_reprocess:
        logger.info(f"Loading preprocessed dataset from {args.processed_data_path}")
        dataset = load_from_disk(args.processed_data_path)
    else:
        logger.info(f"Processing dataset from {args.train_val_path} and {args.test_path}")
        dataset = load_msrvtt(
            train_val_path=args.train_val_path,
            test_path=args.test_path,
            blip_scene_captions=blip_scene_captions,
            max_train_videos=args.max_train_videos,
            seed=args.seed
        )
        dataset.save_to_disk(args.processed_data_path)
    
    logger.info(f"Dataset loaded: {len(dataset['train'])} training, {len(dataset['validation'])} validation, {len(dataset['test'])} test examples")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_input_length, args.max_target_length
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Apply LoRA if selected
    if args.use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning...")
        # Prepare model for LoRA
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "v"]  # For T5, target query and value matrices
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        label_names=["labels"],
    )
    
    # Define data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # Create compute_metrics function with closure for tokenizer
    compute_metrics_with_tokenizer = lambda eval_preds: compute_metrics(eval_preds, tokenizer)
    
    # Define trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_tokenizer,
    )
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Log training metrics
    logger.info(f"Training metrics: {train_result.metrics}")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(
        tokenized_dataset["test"], 
        metric_key_prefix="test"
    )
    
    # Log evaluation metrics
    logger.info(f"Test metrics: {test_results}")
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    
    # Save results in readable format
    results_path = os.path.join(args.output_dir, "test_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
    
    # Generate predictions for test set
    logger.info("Generating predictions for test set...")
    test_predictions = trainer.predict(tokenized_dataset["test"])
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        test_predictions.predictions, 
        skip_special_tokens=True
    )
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, "test_predictions.txt")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for pred in decoded_preds:
            f.write(f"{pred}\n")
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Predictions saved to {predictions_path}")
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
