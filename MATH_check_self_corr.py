import google.generativeai as genai
import json
import os
import re
import argparse
import matplotlib.pyplot as plt

def extract_boxed_content(response):
    """
    Extracts the content between 'boxed{' and '}' from the response string.

    Args:
        response (str): The response string.

    Returns:
        str: The extracted content or an empty string if no match is found.
    """
    start_keyword = "\\boxed{"
    start_index = response.find(start_keyword)
    if start_index == -1:
        return ""

    start_index += len(start_keyword)
    stack = 1
    for i in range(start_index, len(response)):
        if response[i] == '{':
            stack += 1
        elif response[i] == '}':
            stack -= 1
            if stack == 0:
                return response[start_index:i].strip()

    return ""


'''
actual answer
\\boxed{\\left( \\frac{17}{10}, -\\frac{1}{10} \\right)}

first round
boxed{(1.7, -0.1)}
\\boxed{(1.7, -0.1)}

strcmp(chopped(inference), chopped(ground_truth)) = 1
>> this a naive approach

==> Fixes
1) import LaTeX parser to evaluate the expression?
2) specifically prompt the model to give a final numeric answer



'''

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

def plot_correctness_one_round(first_results, next_results, folder_path="./processed_dataset/MATH/train/precalculus", model_name="gemini-1.5-flash-8b", mode="test"):
    # Example usage
    first_results, next_results = MATH_evaluate_accuracy(folder_path)
    print(f"First results: {first_results}")
    print(f"Next results: {next_results}")

    '''
    Results that were plotted and went into the midway report.
    '''
    # first_results = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    # next_results = [[1], [0], [0], [0], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [0], [1], [1], [0], [1], [1], [1], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0]]

    # Flatten next_results to get a single list
    flattened_next_results = [item for sublist in next_results for item in sublist]

    first_accuracy = sum(first_results) / len(first_results)
    second_accuracy = sum(flattened_next_results) / len(flattened_next_results)
    improvement = second_accuracy - first_accuracy
    print(f"First time accuracy: {first_accuracy} \nNext time accuracy: {second_accuracy} \nImprovement: {improvement}")

    # Find indices where first_results == 0 and next_results == 1
    improvement_indices = [i for i, (first, next_) in enumerate(zip(first_results, next_results)) if first == 0 and 1 in next_]

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot first results
    plt.plot(first_results, 'bo', markersize=9, label='First Try Results')

    # Plot next results with larger size and different color
    plt.plot(flattened_next_results, 'ro', label='Second Try Results')

    # Adding labels and title
    plt.xlabel('Question Number')
    plt.ylabel('Correct (1) / Incorrect (0)')
    plt.title('Base Gemini-1.5-Flash-8B Accuracy on MATH Precalculus Dataset')
    plt.legend()

    # Ensure the results directory exists
    results_dir = "results/"
    config_dir = model_name + "/" + mode + "/"
    results_dir = os.path.join(results_dir, config_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    txt_filename = os.path.basename(folder_path) + "_acc_sum.txt" # take last part of the path after last slash
    results_file = os.path.join(results_dir, txt_filename)
    with open(results_file, "w") as file:
        file.write(f"Folder Path: {folder_path}\n")
        file.write(f"First Accuracy: {first_accuracy}\n")
        file.write(f"Second Accuracy: {second_accuracy}\n")
        file.write(f"Improvement: {improvement}\n")
        file.write(f"Improvement Indices: {improvement_indices}\n")

    print(f"Results saved to {results_file}")

    # Save plot to the results directory
    filename = os.path.basename(folder_path) + "_accuracy.png"
    plot_filename = os.path.join(results_dir, filename)
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # Show plot
    # plt.show()
    
'''
Manual run with F5
'''
# first_results, next_results = MATH_evaluate_accuracy("./processed_dataset/MATH/train/precalculus")
# plot_correctness_one_round(first_results, next_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model accuracy and plot results.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing JSON files.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the folder containing JSON files.")
    parser.add_argument("--mode", type=str, required=True, help="Path to the folder containing JSON files.")
    args = parser.parse_args()

    folder_path = args.folder_path
    model_name = args.model_name
    mode = args.mode # train or test, specify as string

    # Call the function to evaluate accuracy and plot results
    first_results, next_results = MATH_evaluate_accuracy(folder_path)
    plot_correctness_one_round(first_results, next_results, folder_path, model_name, mode)