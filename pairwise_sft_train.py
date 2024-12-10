import google.generativeai as genai
import random
import json
import os

def train_with_pairwise_sft(base_model, dataset, epochs=100, learning_rate=0.001):
    """
    Trains the model using pairwise supervised fine-tuning (SFT).

    Args:
        base_model (str): The base Gemini model name.
        dataset (list): A list of data entries, each containing:
                        - `prompt`: Initial question.
                        - `inference_1`: First model inference.
                        - `inference_2`: Second model inference (correction).
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for fine-tuning.

    Returns:
        str: Name of the fine-tuned model.
    """
    # Prepare training data in pairwise format  # This isn't pair-wise
    # use different file
    training_data = [
        {
            "text_input": f"Prompt: {entry['question']}"
            "output": f": {entry['inference_1']}",
            "output": entry["inference_2"]
        }
        for entry in dataset
    ]

    # Create and fine-tune the model
    model_name = f"pairwise-sft-tuned-{random.randint(0, 10000)}"
    operation = genai.create_tuned_model(
        source_model=base_model,
        training_data=training_data,
        id=model_name,
        epoch_count=epochs,
        learning_rate=learning_rate
    )
    operation.result()  # Wait for the fine-tuning process to complete

    print(f"Pairwise SFT training completed. Model name: {model_name}")
    return model_name

'''
example_dataset = [
    {
        "prompt": "What is 3 + 5?",
        "inference_1": "The answer is 10.",
        "inference_2": "The answer is 8."
    },
    {
        "prompt": "Simplify x^2 + 2x + 1.",
        "inference_1": "The simplification is x^2 + x + x + 1.",
        "inference_2": "The simplification is (x + 1)^2."
    }
]
'''

def load_dataset_from_directory(directory_path):
    dataset = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                dataset.extend(data)
    return dataset


if __name__ == "__main__":
    base_model = "gemini-1.5-flash-8b"
    dataset_file = "inferences_0.json"  # Change this to the path of your JSON file
    dataset = load_dataset_from_json(dataset_file)
    
    final_model_name = train_with_pairwise_sft(
        base_model=base_model,
        dataset=dataset,
        epochs=50,
        learning_rate=0.001
    )
    print(f"Final trained model: {final_model_name}")