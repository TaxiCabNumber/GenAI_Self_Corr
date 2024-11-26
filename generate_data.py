import json
from utils import dataloader, load_one_data, generate_inference, save_inferences_to_json, compile_training_dataset
from prompts import initial_prompt, self_correcting_prompt, MATH_initial_prompt, MATH_self_correcting_prompt

# def generate_dataset(dataset_path, model_name, output_path, number_of_rounds, init_prompt=initial_prompt, self_corr_prompt=self_correcting_prompt, dataset_format="json"):
#     """
#     Main routine to handle external datasets, generate inferences, and compile training data.

#     Args:
#         dataset_path (str): Path to the external dataset file.
#         model_name (str): Gemini model name for inference generation.
#         output_path (str): Path to save the processed dataset.
#         dataset_format (str): Format of the input dataset ("json", "csv", or "jsonl").

#     Returns:
#         list: The compiled training dataset.
#     """
#     key_mapping = {
#         "GSM8K": "question",
#         "MATH": "problem"
#     }
#     print(f'dataset_path:{dataset_path}')
#     question_key = None
#     for key in key_mapping:
#         if key in dataset_path:
#             question_key = key_mapping[key]
#             break
#     if question_key is None:
#         raise ValueError("Dataset path must contain either 'GSM8K' or 'MATH'.")

#     # Generate inferences
#     inferences = []
#     for entry in load_one_data(dataset_path, dataset_format):
#         question = entry[question_key]
#         print(f'Processing question: {question}')

#         # First query with initial prompt
#         inference = generate_inference(f"{init_prompt}\n{question}", model_name)
#         inference["ground_truth"] = entry.get("answer", None)  # Include ground truth if available
#         inferences.append(inference)

#         # Subsequent queries with self-correcting prompt
#         for _ in range(number_of_rounds):  # Adjust the range for more iterations if needed
#             inference = generate_inference(f"{inference['inference']}\n{self_corr_prompt}", model_name)
#             inference["ground_truth"] = entry.get("answer", None)  # Include ground truth if available
#             inferences.append(inference)

#         # Save inferences to a structured JSON file after each entry
#         save_inferences_to_json(inferences, f"{output_path}_inferences.json")
#         inferences = []  # Clear inferences after saving

#     # Compile the training dataset
#     training_dataset = compile_training_dataset(load_one_data(dataset_path, dataset_format), inferences)

#     # Save the compiled dataset
#     with open(f"{output_path}_compiled.json", "w") as compiled_file:
#         json.dump(training_dataset, compiled_file, indent=4)

#     print(f"Processed dataset saved at {output_path}_compiled.json.")
#     return training_dataset

# # Example usage
# if __name__ == "__main__":
#     dataset_path = "./data/gsm8k_train.jsonl"
#     MODEL_NAME = "gemini-1.5-flash-8b"

#     # Example usage with a JSONL dataset
#     gsm8k_training_dataset = generate_dataset(
#         dataset_path=dataset_path,
#         model_name=MODEL_NAME,
#         output_path="processed_dataset",
#         number_of_rounds=1,
#         init_prompt=initial_prompt,
#         self_corr_prompt=self_correcting_prompt,
#         dataset_format="jsonl"
#     )

#     dataset_path = "./data/MATH/train/precalculus"
#     MATH_verification_dataset = generate_dataset(
#         dataset_path=dataset_path,
#         model_name=MODEL_NAME,
#         output_path="processed_dataset",
#         number_of_rounds=1,
#         init_prompt=MATH_initial_prompt,
#         self_corr_prompt=MATH_self_correcting_prompt,
#         dataset_format="json"
#     )









def generate_dataset(dataset_path, model_name, output_path, number_of_rounds, init_prompt=initial_prompt, self_corr_prompt=self_correcting_prompt, dataset_format="json"):
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
    # make a dataloader to only take in 4 examples at once for GSM8K, 2 for MATH
    big_dataset = dataloader(dataset_path, dataset_format=dataset_format)

    key_mapping = {
        "GSM8K": "question",
        "MATH": "problem"
    }
    question_key = None
    for key in key_mapping:
        if key in dataset_path:
            question_key = key_mapping[key]
            break
    if question_key is None:
        raise ValueError("Dataset path must contain either 'GSM8K' or 'MATH'.")

    # Generate inferences
    inferences = []
    # if big_dataset is a list of json files as a jsonl file, then we have to iterate over each list

    for dataset in big_dataset:
        for entry in dataset:
            question = entry[question_key]
            # print(f'Processing question: {question}')

            # First query with initial prompt
            inference = generate_inference(f"{init_prompt}\n{question}", model_name)
            inference["ground_truth"] = entry.get("answer", None)  # Include ground truth if available
            inferences.append(inference)

            # Subsequent queries with self-correcting prompt
            for _ in range(number_of_rounds):  # Adjust the range for more iterations if needed
                inference = generate_inference(f"{inference['inference']}\n{self_corr_prompt}", model_name) # uses (n-1)th inference to generate nth
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


data_path = "./data/GSM8K_train.jsonl"
MODEL_NAME = "gemini-1.5-flash-8b"

# Example usage with a JSONL dataset
gsm8k_training_dataset = generate_dataset(
    dataset_path=data_path,
    model_name=MODEL_NAME,
    output_path="processed_dataset",
    number_of_rounds=1,
    init_prompt=initial_prompt,
    self_corr_prompt=self_correcting_prompt,
    dataset_format="jsonl"
)

data_path = "./data/MATH/train/precalculus"
MATH_verfication_dataset = generate_dataset(
    dataset_path=data_path,
    model_name=MODEL_NAME,
    output_path="processed_dataset",
    number_of_rounds=1,
    init_prompt=MATH_initial_prompt,
    self_corr_prompt=MATH_self_correcting_prompt,
    dataset_format="json"
)

# # Example usage with a CSV dataset in TheVault format
# training_dataset = generate_dataset(
#     dataset_path="thevault.csv",
#     model_name="gemini-1.5-flash",
#     output_path="processed_dataset",
#     number_of_rounds=1,
#     dataset_format="csv"
# )
