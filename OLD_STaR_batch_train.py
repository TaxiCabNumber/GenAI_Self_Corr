import os
import json
import argparse

def extract_content_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        extracted_data = []
        for entry in data:
            if "question" in entry and "inference" in entry:
                extracted_data.append({
                    "text_input": entry["question"],
                    "output": entry["inference"]
                })
        return extracted_data

def process_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for folder_name in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder_name)
        if os.path.isdir(folder_path):
            all_extracted_data = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(folder_path, file_name)
                    extracted_data = extract_content_from_json(file_path)
                    
                    ''' run training loop here'''
                    # Use this as the training data
                    all_extracted_data.extend(extracted_data) 


            
            # output_file_path = os.path.join(output_directory, f"{folder_name}_extracted.json")
            # with open(output_file_path, 'w') as output_file:
            #     json.dump(all_extracted_data, output_file, indent=4)
            # print(f"Extracted data saved to {output_file_path}")

def main(input_directory, output_directory):
    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of folders containing JSON files.")
    parser.add_argument("input_directory", type=str, help="Path to the directory containing folders with JSON files.")
    parser.add_argument("output_directory", type=str, help="Path to the directory to save the extracted JSON files.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_directory):
        print(f"Error: The directory {args.input_directory} does not exist.")
        exit(1)
    
    main(args.input_directory, args.output_directory)


# import google.generativeai as genai
# import random

# def train_with_star(base_model, original_data, iterations=3, epochs=100, learning_rate=0.001):
#     """
#     Trains the model using the STaR self-taught reasoning approach.

#     Args:
#         base_model (str): The base Gemini model name.
#         original_data (list): A list of original question-answer pairs.
#         iterations (int): Number of iterative self-taught reasoning cycles.
#         epochs (int): Number of epochs for training in each iteration.
#         learning_rate (float): Learning rate for optimization.

#     Returns:
#         str: The final trained model name after iterative STaR training.
#     """
#     # Initialize training data with original dataset
#     current_training_data = [{"text_input": item["question"], "output": item["answer"]} for item in original_data]
#     model_name = base_model

#     for iteration in range(iterations):
#         print(f"Starting STaR iteration {iteration + 1}/{iterations}...")

#         # Generate reasoning traces using the current model
#         reasoning_traces = []
#         model = genai.GenerativeModel(model_name=model_name)

#         for item in original_data:
#             response = model.generate_content(item["question"])
#             reasoning_traces.append({
#                 "question": item["question"],
#                 "generated_trace": response.text,
#                 "ground_truth": item["answer"]
#             })

#         # Filter traces that align with ground truth (successful reasoning examples)
#         successful_traces = [
#             {"text_input": trace["question"], "output": trace["generated_trace"]}
#             for trace in reasoning_traces if trace["generated_trace"].strip() == trace["ground_truth"].strip()
#         ]

#         print(f"Filtered {len(successful_traces)} successful reasoning traces.")

#         # Update training data by combining original data and successful reasoning traces
#         current_training_data.extend(successful_traces)

#         # Re-train the model on the updated dataset
#         model_name = f"star-tuned-{random.randint(0, 10000)}"
#         operation = genai.create_tuned_model(
#             source_model=base_model if iteration == 0 else model_name,
#             training_data=current_training_data,
#             id=model_name,
#             epoch_count=epochs,
#             learning_rate=learning_rate
#         )
#         operation.result()  # Wait for the training process to complete

#         print(f"Completed iteration {iteration + 1}. Model {model_name} created.")

#     return model_name


# # Example original dataset
# original_data = [
#     {"question": "What is 2 + 2?", "answer": "4"},
#     {"question": "What is the derivative of x^2?", "answer": "2x"}
# ]

# # Train using STaR
# final_model = train_with_star(
#     base_model="gemini-1.5-flash",
#     original_data=original_data,
#     iterations=3,
#     epochs=50,
#     learning_rate=0.001
# )
# print(f"Final trained model: {final_model}")
