import google.generativeai as genai
import random
import json
import re
from generate_data import generate_dataset
from prompts import initial_prompt, self_correcting_prompt 
from key import google_api_key 
import csv
import os

'''
The code uses the following outline generate a new dataset to train self correction 

1. Generate Inference
Develop a method that queries a math or reasoning problem dataset (e.g., MATH or TheVault) and generates inferences using an LLM.

2. Store Inference in JSON
Create a structured JSON format to save these inferences, ensuring compatibility with the original dataset structure.

3. Compile New Training Dataset
Combine the original dataset and self-generated inferences into a single structured training dataset for further training.

4. Implement REINFORCE-Based Algorithm
Train the LLM using the REINFORCE algorithm to optimize self-correction behavior across multiple attempts.

5. Accuracy Check Method
Validate the model's final inferences against actual answers to measure accuracy.

'''
dataset_path = "./generated_chains/MATH/train"
model_name = "gemini-1.5-flash-8b"
output_path = "gsm8k_processed"


def main():
    # Define paths and model name
    dataset_path = "gsm8k.json"
    model_name = "gemini-1.5-flash-8b"
    output_path = "gsm8k_processed"
    
    ## TODO
    evaluation_data_path = "evaluation_data.json" 

    # Step 1: Generate Inference
    # generate_dataset(dataset_path, model_name, output_path, dataset_format="json")

    # Load the generated dataset
    with open(output_path, "r") as file:
        training_data = json.load(file)

    # Step 2: Train with REINFORCE
    trained_model_name = train_with_reinforce(model_name, training_data)

    # Load the evaluation data
    with open(evaluation_data_path, "r") as file:
        evaluation_data = json.load(file)

    # Step 3: Evaluate Accuracy
    accuracy = evaluate_accuracy(trained_model_name, evaluation_data)
    print(f"Model accuracy: {accuracy * 100:.2f}%")


def train_with_reinforce(base_model, training_data, epochs=100, learning_rate=0.001):
    """
    Trains the Gemini model using the REINFORCE algorithm.

    Args:
        base_model (str): The base Gemini model name.
        training_data (list): The training data for fine-tuning.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for optimization.

    Returns:
        str: Name of the trained model.
    """
    model_name = f"reinforce-tuned-{random.randint(0, 10000)}"
    operation = genai.create_tuned_model(
        source_model=base_model,
        training_data=[
            {"text_input": entry["original_question"], "output": entry["inference"]}
            for entry in training_data
        ],
        id=model_name,
        epoch_count=epochs,
        learning_rate=learning_rate
    )
    operation.result()  # Wait for training to complete
    return model_name


# def evaluate_accuracy(model_name, evaluation_data):
#     """
#     Evaluates the accuracy of the model's inferences.

#     Args:
#         model_name (str): The Gemini model name.
#         evaluation_data (list): A list of evaluation questions with answers.

#     Returns:
#         float: Accuracy of the model's responses.
#     """
#     model = genai.GenerativeModel(model_name=model_name)
#     correct_count = 0

#     for item in evaluation_data:
#         response = model.generate_content(item["question"]).text
#         if response.strip() == item["answer"].strip():
#             correct_count += 1

#     return correct_count / len(evaluation_data)


def MATH_evaluate_accuracy(model_name, evaluation_data):
    """
    Evaluates the accuracy of the model's inferences.

    Args:
        model_name (str): The Gemini model name.
        evaluation_data (list): A list of evaluation questions with answers.

    Returns:
        float: Accuracy of the model's responses.
    """
    model = genai.GenerativeModel(model_name=model_name)
    correct_count = 0

    # \\boxed{2008}$.
    for item in evaluation_data:
        response = model.generate_content(item["question"]).text
        # Extract the part of the response after "Answer" or "... Final Answer:"
        if "boxed:" in response:
            extracted_response = response.split("boxed{")[-1].strip()
        elif "Answer" in response:
            extracted_response = response.split("Answer")[-1].strip()
        elif "answer" in response:
            extracted_response = response.split("answer")[-1].strip()
        elif "nswer:" in response:
            extracted_response = response.split("nswer:")[-1].strip()
        else:
            extracted_response = response.strip()

        solution = item["solution"].strip()
        extracted_solution = solution.split("boxed{")[-1].strip()
        if extracted_response == extracted_solution:
            correct_count += 1

    return correct_count / len(evaluation_data)


def evaluate_accuracy(model_name, evaluation_data):
    """
    Evaluates the accuracy of the model's inferences.

    Args:
        model_name (str): The Gemini model name.
        evaluation_data (list): A list of evaluation questions with answers.

    Returns:
        float: Accuracy of the model's responses.
    """
    model = genai.GenerativeModel(model_name=model_name)
    correct_count = 0

    for item in evaluation_data:
        response = model.generate_content(item["question"]).text
        # Extract the part of the response after "Answer" or "... Final Answer:"
        if "Final Answer:" in response:
            extracted_response = response.split("Final Answer:")[-1].strip()
        elif "Answer" in response:
            extracted_response = response.split("Answer")[-1].strip()
        elif "answer" in response:
            extracted_response = response.split("Answer")[-1].strip()
        elif "answer:" in response:
            extracted_response = response.split("answer:")[-1].strip()
        else:
            extracted_response = response.strip()

        if extracted_response == item["answer"].strip():
            correct_count += 1

    return correct_count / len(evaluation_data)


if __name__ == "__main__":
    main()