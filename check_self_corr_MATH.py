import google.generativeai as genai
import random
import json
import re
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
    match = re.search(r'boxed\{(.*?)\}', response)
    if match:
        return match.group(1).strip()
    return ""

# Example usage
response = "Some text before boxed{desired content} some text after"
stripped_response = response.strip()
extracted_content = extract_boxed_content(stripped_response)
print(extracted_content)  # Output: desired content


def MATH_evaluate_accuracy(model_name, evaluation_data):
    """
    Evaluates the accuracy of the model's inferences.

    Args:
        model_name (str): The Gemini model name.
        evaluation_data (list): A l.

    Returns:
        float: Accuracy of the model's responses.
    """
    correct_count = 0

    # \\boxed{2008}$.
    for item in evaluation_data:
        response = evaluation_data(item["problem"]).text
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

