import google.generativeai as genai
import json
import re
import os
import matplotlib.pyplot as plt
from generate_data import generate_dataset
from prompts import initial_prompt, self_correcting_prompt 

import re

def extract_boxed_content(response):
    """
    Extracts the content between 'boxed{' and '}' from the response string.

    Args:
        response (str): The response string.

    Returns:
        str: The extracted content or an empty string if no match is found.
    """
    match = re.search(r'boxed\{(.*?)\$', response)
    if match:
        return match.group(1).strip()
    return ""

def MATH_evaluate_accuracy(folder_path):
    """
    Evaluates the accuracy of the model's inferences from JSON files in a folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        tuple: A tuple containing two lists - first_results and next_results.
    """
    first_results = []
    next_results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

            first_time_result = 0
            next_time_results = []

            # Process the first JSON object
            first_inference = extract_boxed_content(data[0]["inference"])
            first_ground_truth = extract_boxed_content(data[0]["ground_truth"])
            print(f"first_inference: {first_inference} \n first_ground_truth: {first_ground_truth}")
            if first_inference == first_ground_truth:
                first_time_result = 1
            first_results.append(first_time_result)

            # Process the remaining JSON objects
            for item in data[1:]:
                next_inference = extract_boxed_content(item["inference"])
                next_ground_truth = extract_boxed_content(item["ground_truth"])
                print(f"next_inference: {next_inference} \n next_ground_truth: {next_ground_truth}")
                next_time_result = 1 if next_inference == next_ground_truth else 0
                next_time_results.append(next_time_result)

            next_results.append(next_time_results)

    return first_results, next_results

# Example usage
folder_path = "./processed_dataset/MATH/train/precalculus"
first_results, next_results = MATH_evaluate_accuracy(folder_path)
print(f"First results: {first_results}")
print(f"Next results: {next_results}")

# first_results = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
# next_results = [[1], [0], [0], [0], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [0], [1], [1], [0], [1], [1], [1], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]]

# Flatten next_results to get a single list
flattened_next_results = [item for sublist in next_results for item in sublist]

# Plotting
plt.figure(figsize=(10, 6))

# Plot first results
plt.plot(first_results, 'bo', label='First Time Results')

# Plot next results with larger size and different color
plt.plot(flattened_next_results, 'ro', markersize=9, label='Next Time Results')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Correct (1) / Incorrect (0)')
plt.title('Model Accuracy: First Time vs Next Time')
plt.legend()

# Show plot
plt.show()