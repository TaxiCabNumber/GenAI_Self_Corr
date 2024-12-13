import google.generativeai as genai
import json
import os
import re
import argparse
import matplotlib.pyplot as plt

# def extract_number_after_pounds(response): # with dollar sign
#     """
#     Extracts the number after 'I hope it is correct' or '####' from the response string.

#     Args:
#         response (str): The response string.

#     Returns:
#         str: The extracted number or an empty string if no match is found.
#     """
#     match_correct = re.search(r'I hope it is correct\s*:\s*####\s*(\$?\d+)', response)
#     match_pounds = re.search(r'####\s*(\$?\d+)', response)

#     if match_correct and match_pounds:
#         if match_correct.group(1) != match_pounds.group(1):
#             return match_correct.group(1)
#         return match_correct.group(1)
#     elif match_correct:
#         return match_correct.group(1)
#     elif match_pounds:
#         return match_pounds.group(1)
#     return ""


def extract_number_after_pounds(response):
    """
    Extracts the number after 'I hope it is correct' or '####' from the response string.

    Args:
        response (str): The response string.

    Returns:
        str: The extracted number or an empty string if no match is found.
    """
    last_correct_index = response.rfind('I hope it is correct')
    last_pounds_index = response.rfind('####')

    if last_correct_index != -1 and (last_correct_index > last_pounds_index or last_pounds_index == -1):
        split_part = response[last_correct_index + len('I hope it is correct'):]
    elif last_pounds_index != -1:
        split_part = response[last_pounds_index + len('####'):]
    else:
        return ""

    # Extract numbers from the split part
    match = re.search(r'(\$?\d+)', split_part)
    if match:
        return match.group(1)
    return ""

def GSM8K_evaluate_accuracy(folder_path):
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
            first_inference = extract_number_after_pounds(data[0]["inference"])
            first_ground_truth = extract_number_after_pounds(data[0]["ground_truth"])
            print(f"first_inference: {first_inference} \n first_ground_truth: {first_ground_truth}")
            if first_inference == first_ground_truth:
                first_time_result = 1
            first_results.append(first_time_result)

            # Process the remaining JSON objects
            for item in data[1:]:
                next_inference = extract_number_after_pounds(item["inference"])
                next_ground_truth = extract_number_after_pounds(item["ground_truth"])
                print(f"next_inference: {next_inference} \n next_ground_truth: {next_ground_truth}")
                next_time_result = 1 if next_inference == next_ground_truth else 0
                next_time_results.append(next_time_result)

            next_results.append(next_time_results)

    return first_results, next_results

def plot_correctness_one_round(first_results, next_results, folder_path="./processed_dataset/GSM8K/train", model_name="gemini-1.5-flash-8b", mode="test"):
    # Example usage
    first_results, next_results = GSM8K_evaluate_accuracy(folder_path)
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

    # Return the ratio of improvement
    print("Pairwise generation statistics")
    incorrect_indices = [i for i, (first, next_) in enumerate(zip(first_results, next_results)) if first == 0 and 0 in next_]
    degradation_indices = [i for i, (first, next_) in enumerate(zip(first_results, next_results)) if first == 1 and 0 in next_]
    correct_indices = [i for i, (first, next_) in enumerate(zip(first_results, next_results)) if first == 1 and 1 in next_]
    
    divisor = len(first_results)
    print(f'GSM8K {mode} split STaR generation for {model_name}')
    print(f"(1,1) Correct: {100*len(correct_indices) / divisor}%")
    print(f"(0,0) Incorrect: {100*len(incorrect_indices) / divisor}%")
    print(f"(1,0) Degradation: {100*len(degradation_indices) / divisor}%")
    print(f"(0,1) Improvement: {100*len(improvement_indices) / divisor}%")


    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot first results
    plt.plot(first_results, 'bo', markersize=9, label='First Try Results')

    # Plot next results with larger size and different color
    plt.plot(flattened_next_results, 'ro', label='Second Try Results')

    # Adding labels and title
    plt.xlabel('Question Number')
    plt.ylabel('Correct (1) / Incorrect (0)')
    plt.title('Base Gemini-1.5-Flash-8B Accuracy on GSM8K Dataset')
    plt.legend()

    # Ensure the results directory exists
    if "MATH" in folder_path:
        dataset = "MATH"
    else:
        dataset = "GSM8K"
    results_dir = "results/"
    config_dir = model_name + "/" + dataset + "/" + mode + "/"
    results_dir = os.path.join(results_dir, config_dir) # does not automatically adds slash
    os.makedirs(results_dir, exist_ok=True) # has to be called last to verify its existence
    
    txt_filename = os.path.basename(folder_path) + "_acc_sum.txt" # take last part of the path after last slash
    results_file = os.path.join(results_dir, txt_filename) # can only write to existing directory
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
# first_results, next_results = GSM8K_evaluate_accuracy("./data/GSM8K_train.jsonl")
# plot_correctness_one_round(first_results, next_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model accuracy and plot results.")
    parser.add_argument("--folder_path", type=str, default="./generated_chains/models/gemini-1.5-flash-001-tuning/GSM8K/test", required=False, help="Path to the folder containing JSON files.")
    parser.add_argument("--model_name", type=str, default="models/gemini-1.5-flash-001-tuning", required=False, help="Path to the folder containing JSON files.")
    parser.add_argument("--mode", type=str, default="test", required=False, help="Path to the folder containing JSON files.")
    args = parser.parse_args()

    folder_path = args.folder_path
    model_name = args.model_name
    mode = args.mode # train or test, specify as string

    # Call the function to evaluate accuracy and plot results
    first_results, next_results = GSM8K_evaluate_accuracy(folder_path)
    plot_correctness_one_round(first_results, next_results, folder_path, model_name, mode)