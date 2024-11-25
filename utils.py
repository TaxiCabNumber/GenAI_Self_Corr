'''
This file contains utility functions for integrating external datasets for generating the finetuning dataset
'''
import google.generativeai as genai
import json
import csv
from key import google_api_key 
import random

def load_dataset(dataset_path, dataset_format="json"):
    """
    Loads a dataset from the given path and format.

    Args:
        dataset_path (str): Path to the dataset file.
        dataset_format (str): Format of the dataset ("json", "csv", or "jsonl").

    Returns:
        list: A list of dictionary entries with "question" and "answer" fields.
    """
    dataset = []
    if dataset_format == "json":
        with open(dataset_path, "r") as json_file:
            dataset = json.load(json_file)
    elif dataset_format == "csv":
        with open(dataset_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            dataset = [{"question": row["question"], "answer": row.get("answer")} for row in reader]
    elif dataset_format == "jsonl":
        with open(dataset_path, "r") as jsonl_file:
            dataset = [json.loads(line) for line in jsonl_file]
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    print(f"Loaded {len(dataset)} entries from {dataset_path}.")
    return dataset


def dataloader(dataset_path, dataset_format="jsonl", batch_size=32):
    """
    Loads a dataset and returns it in batches.

    Args:
        dataset_path (str): Path to the dataset file.
        dataset_format (str): Format of the dataset ("json", "csv", or "jsonl").
        batch_size (int): Number of entries per batch.

    Returns:
        list: A list of batches, where each batch is a list of dictionary entries.
    """
    dataset = load_dataset(dataset_path, dataset_format)
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    return batches


def generate_inference(question, model_name):
    """
    Generates an inference from the Gemini model for the given question.

    Args:
        question (str): The question for which the inference is generated.
        model_name (str): The Gemini model name.

    Returns:
        dict: A dictionary containing the question and its generated response.
    """
    genai.configure(api_key=google_api_key) # api_key = userdata.get('GOOGLE_API_KEY')
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(question)

    # This may clutter your computer.
    print(f"question: {question} \n inference: {response.text}")
    return {
        "question": question,
        "inference": response.text
    }


# def save_inferences_to_json(inference, file_path):
#     """
#     Saves an inference to a JSON file.

#     Args:
#         inference (dict): The inference to save.
#         file_path (str): The file path for the JSON output.
#     """
#     with open(file_path, "a") as json_file:
#         json.dump(inference, json_file, indent=4)
#         json_file.write(",\n")  # Append new entries cleanly
#     print(f"Inferences saved to {file_path}.")

def save_inferences_to_json(inferences, file_path):
    """
    Saves inferences to a JSON file.

    Args:
        inferences (list): The inferences to save.
        file_path (str): The file path for the JSON output.
    """
    with open(file_path, "w") as json_file:
        json.dump(inferences, json_file, indent=4)
    print(f"Inferences saved to {file_path}.")


def compile_training_dataset(original_data, inferences):
    """
    Combines original dataset with generated inferences.

    Args:
        original_data (list): A list of original question-answer pairs.
        inferences (list): A list of self-generated inferences.

    Returns:
        list: A new dataset combining both original and self-corrected entries.
    """
    combined_dataset = []
    for original, inference in zip(original_data, inferences):
        combined_dataset.append({
            "original_question": original.get["question"],
            "original_answer": original.get["answer"],
            "inference": inference.get["inference"]
        })
    return combined_dataset