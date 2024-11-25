import google.generativeai as genai
import json
from utils import load_dataset, generate_inference, save_inferences_to_json, compile_training_dataset

def generate_dataset(dataset_path, model_name, output_path, dataset_format="json"):
    """
    Main routine to handle external datasets, generate inferences, and compile training data.

    Args:
        dataset_path (str): Path to the external dataset file.
        model_name (str): Gemini model name for inference generation.
        output_path (str): Path to save the processed dataset.
        dataset_format (str): Format of the input dataset ("json" or "csv").

    Returns:
        list: The compiled training dataset.
    """
    # Load the dataset
    dataset = load_dataset(dataset_path, dataset_format)

    # Generate inferences
    inferences = []
    for entry in dataset:
        inference = generate_inference(entry["question"], model_name)
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


# Example usage with a JSON dataset in GSM8K format
training_dataset = generate_dataset(
    dataset_path="gsm8k.json",
    model_name="gemini-flash-1.0",
    output_path="processed_dataset",
    dataset_format="json"
)

# Example usage with a CSV dataset in TheVault format
training_dataset = generate_dataset(
    dataset_path="thevault.csv",
    model_name="gemini-flash-1.0",
    output_path="processed_dataset",
    dataset_format="csv"
)
