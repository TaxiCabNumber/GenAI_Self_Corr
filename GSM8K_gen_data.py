import json
import os
import pickle
import argparse
from utils import dataloader, load_one_data, generate_inference, save_inferences_to_json, compile_training_dataset
from prompts import GSM8K_initial_prompt, GSM8K_self_correcting_prompt

def save_idx(output_path, idx):
    with open(os.path.join(output_path, "current_idx.pkl"), "wb") as f:
        pickle.dump(idx, f)

def load_idx(output_path):
    idx_file = os.path.join(output_path, "current_idx.pkl")
    if os.path.exists(idx_file):
        with open(idx_file, "rb") as f:
            return pickle.load(f)
    return 0

def generate_dataset(dataset_path, model_name, output_path, number_of_rounds, init_prompt=GSM8K_initial_prompt, self_corr_prompt=GSM8K_self_correcting_prompt, dataset_format="jsonl"):
    """
    Main routine to handle external datasets, generate inferences, and compile training data.

    Args:
        dataset_path (str): Path to the external dataset file.
        model_name (str): Gemini model name for inference generation.
        output_path (str): Path to save the processed dataset.
        dataset_format (str): Format of the input dataset ("json", "csv", or "jsonl").

    Returns:
        list: The compiled training dataset.
    """
    key_mapping = {
        "GSM8K": "question",
        "MATH": "problem"
    }
    print(f'dataset_path:{dataset_path}')
    question_key = None
    for key in key_mapping:
        if key in dataset_path:
            question_key = key_mapping[key]
            break
    if question_key is None:
        raise ValueError("Dataset path must contain either 'GSM8K' or 'MATH'.")

    soln_mapping = {
        "GSM8K": "answer",
        "MATH": "solution"
    }
    answer_key = None
    for key in key_mapping:
        if key in dataset_path:
            answer_key = soln_mapping[key]
            break
    if answer_key is None:
        raise ValueError("Dataset path must contain either 'GSM8K' or 'MATH'.")
    
    os.makedirs(output_path, exist_ok=True)

    # Load the last processed index
    idx = load_idx(output_path)

    if idx > 3000:
        print("Already processed 100 entries. Skipping the dataset.")
        return
    
    # Generate inferences
    inferences = []
    for current_idx, entry in enumerate(load_one_data(dataset_path, dataset_format)):
        if current_idx < idx:
            continue  # Skip already processed entries

        question = entry[question_key]
        print(f'Processing question: {question}')

        # First query with initial prompt
        inference = generate_inference(f"{init_prompt}\n{question}", model_name)
        inference["ground_truth"] = entry.get(answer_key, None)  # Include ground truth if available
        inferences.append(inference)

        # Subsequent queries with self-correcting prompt
        for _ in range(number_of_rounds):  # Adjust the range for more iterations if needed
            inference = generate_inference(f"{inference['inference']}\n{self_corr_prompt}", model_name)
            inference["ground_truth"] = entry.get(answer_key, None)  # Include ground truth if available
            inferences.append(inference)

        # Save inferences to a structured JSON file after each entry
        unique_output_path = os.path.join(output_path, f"inferences_{current_idx}.json")
        save_inferences_to_json(inferences, unique_output_path)
        inferences = []  # Clear inferences after saving

        # Save the current index
        save_idx(output_path, current_idx + 1)

    # Compile the training dataset
    training_dataset = compile_training_dataset(load_one_data(dataset_path, dataset_format), inferences)

    # Save the compiled dataset
    compiled_output_path = os.path.join(output_path, "compiled_dataset.json")
    with open(compiled_output_path, "w") as compiled_file:
        json.dump(training_dataset, compiled_file, indent=4)

    print(f"Processed dataset saved at {compiled_output_path}.")
    return training_dataset

# Example usage

if __name__ == "__main__":
    MODEL_NAME = "gemini-1.5-flash-8b"
    dataset_path = "./data/GSM8K_train.jsonl"
    output_path = f"{MODEL_NAME}_generated_dataset/GSM8K"

    parser = argparse.ArgumentParser(description="Generate dataset with inferences.")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, required=False, help="Path to the external dataset file.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, required=False, help="Gemini model name for inference generation.")
    parser.add_argument("--output_path", type=str, default=output_path, required=False, help="Path to save the processed dataset.")
    parser.add_argument("--number_of_rounds", type=int, default=1, help="Number of self-correcting rounds.")
    parser.add_argument("--init_prompt", type=str, default=GSM8K_initial_prompt, help="Initial prompt to use for generating inferences.")
    parser.add_argument("--self_corr_prompt", type=str, default=GSM8K_self_correcting_prompt, help="Self-correcting prompt to use for subsequent inferences.")
    parser.add_argument("--dataset_format", type=str, default="jsonl", help="Format of the input dataset ('json', 'csv', or 'jsonl').")

    args = parser.parse_args()

    generate_dataset(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_path=args.output_path,
        number_of_rounds=args.number_of_rounds,
        init_prompt=args.init_prompt,
        self_corr_prompt=args.self_corr_prompt,
        dataset_format=args.dataset_format
    )