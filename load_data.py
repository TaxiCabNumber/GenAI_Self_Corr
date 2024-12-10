import google.generativeai as genai
from load_data import load_inferences_from_directory
from key import google_api_key
import random
import json
import os
import re


def load_inferences_from_directory(directory_path):
    """
    Loads all JSON files from a directory and extracts "question" and "inference" values into a list.

    Args:
        directory_path (str): Path to the directory containing JSON files.

    Returns:
        list: List of dictionaries with "question" and "inference" keys.
    """
    training_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file) # per file 
                for entry in data:
                    if isinstance(entry.get("question"), str) and isinstance(entry.get("inference"), str):
                        training_data.append({
                            "question": entry["question"],
                            "inference": entry["inference"]
                        })
    return training_data # this is all files
'''
    example_data=[
        {
             'question': '"You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form 'Final Answer: \\\\boxed{$answer$}. I hope it is correct.'\nMegan has lost Fatima's phone number. Megan knows that the first three digits are either 296 or 299. The remaining four digits are 0, 1, 6 and 7, but she isn't sure of the order of these digits. If Megan randomly dials a seven-digit number that meets these conditions, what is the probability that she dials Fatima's correct number? Express your answer as a common fraction."',
             'inference': '"There are 2 possible first three digits (296 or 299).\nFor each choice of first three digits, there are 4! = 24 possible orderings of the remaining four digits (0, 1, 6, 7).\nTotal possible numbers = 2 * 24 = 48\nThe probability that Megan dials Fatima's correct number is 1/48.\nFinal Answer: $\\boxed{\\frac{1}{48}}$\n"'},
             {
             'question': '...',
             'inference': '...',
        },{
             'question': '...',
             'inference': '...',
        },{
             'question': '...',
             'inference': '...',
        }]
'''

'''
then based on how many rounds N you self correct, parse
K entries at a time, where K = len(example_data) // N
From these K entries, take the first one as the initial prompt-response pair,
and the rest as self corrections.

for file in directory

'''

dataset_directory = "./generated_dataset/MATH/train"
model_name = "gemini-1.5-flash-8b"
output_path = "MATH_processed"

def main():
    # Define paths and model name
    dataset_directory = "./generated_dataset/MATH/train"
    model_name = "gemini-1.5-flash-8b"
    output_path = "MATH_processed"
    
    # Load the generated dataset
    training_data = load_inferences_from_directory(dataset_directory)

    # Step 2: Train with REINFORCE
    trained_model_name = train_with_reinforce(model_name, training_data)

    # Load the evaluation data
    evaluation_data_path = "evaluation_data.json"
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
            {"text_input": entry["question"], "output": entry["inference"]}
            for entry in training_data
        ],
        id=model_name,
        epoch_count=epochs,
        learning_rate=learning_rate
    )
    operation.result()  # Wait for training to complete
    return model_name



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