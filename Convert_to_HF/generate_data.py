import google.generativeai as genai
import json
from utils import dataloader, generate_inference, save_inferences_to_json, compile_training_dataset
from prompts import initial_prompt, self_correcting_prompt

def generate_dataset(dataset_path, model_name, output_path, number_of_rounds, dataset_format="json"):
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
    # Load the dataset
    big_dataset = dataloader(dataset_path, dataset_format='jsonl', batch_size=4)

    # make a dataloader to only take in 4 examples at once
    # dataloader = DataLoader(dataset, batch_size=4)

    # Generate inferences
    inferences = []
    for dataset in big_dataset:
        for entry in dataset:
            question = entry["question"]
            print(f'Processing question: {question}')
            
            # First query with initial prompt
            inference = generate_inference(f"{initial_prompt}\n{question}", model_name)
            inference["ground_truth"] = entry.get("answer", None)  # Include ground truth if available
            inferences.append(inference)
            
            # Subsequent queries with self-correcting prompt
            for _ in range(number_of_rounds):  # Adjust the range for more iterations if needed
                inference = generate_inference(f"{self_correcting_prompt}\n{inference['inference']}", model_name)
                inference["ground_truth"] = entry.get("answer", None)  # Include ground truth if available
                inferences.append(inference)

    # Save inferences to a structured JSON file
    save_inferences_to_json(inferences, f"{output_path}_inferences.json")

    # Compile the training dataset
    training_dataset = compile_training_dataset(dataset, inferences)

    # Save the compiled dataset
    with open(f"{output_path}_compiled.json", "w") as compiled_file:
        json.dump(training_dataset, compiled_file, indent=4)

    print(f"Processed dataset saved at {output_path}_compiled.json.")
    return training_dataset


data_path = "./data/train.jsonl"
MODEL_NAME = "gemini-1.5-flash-8b"

# Example usage with a JSONL dataset
training_dataset = generate_dataset(
    dataset_path=data_path,
    model_name=MODEL_NAME,
    output_path="processed_dataset",
    number_of_rounds=1,
    dataset_format="jsonl"
)

# Example usage with a CSV dataset in TheVault format
# training_dataset = generate_dataset(
#     dataset_path="thevault.csv",
#     model_name="gemini-1.5-flash",
#     output_path="processed_dataset",
#     number_of_rounds=1,
#     dataset_format="csv"
# )
