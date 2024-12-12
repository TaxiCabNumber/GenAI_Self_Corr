import os
import json
import argparse
import google.generativeai as genai
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


base_name = "models/gemini-1.5-flash-001-tuning"
tuned_name = "math-train"
id = 1 # increment this number for each new model

from key import google_api_key 
genai.configure(api_key=google_api_key) # api_key = userdata.get('GOOGLE_API_KEY')
initialize_model = True

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
    global initialize_model  # Add this line
    global id
    global tuned_name
    global base_name 
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

                    if extracted_data and initialize_model:
                        operation = genai.create_tuned_model(
                        display_name="Flash 1.5 8B MATH",
                        source_model=base_name,
                        id=f"{tuned_name}-{id}",
                        epoch_count=5,
                        batch_size=2,
                        learning_rate=0.001,
                        training_data=extracted_data,
                        )
                        initialize_model = False # toggle to false
                        print(f"generated {tuned_name}-{id}")
                        for status in operation.wait_bar():
                            time.sleep(10)

                    elif extracted_data: # use your new model
                        operation = genai.create_tuned_model(
                        display_name="Flash 1.5 8B MATH",
                        source_model=f"tunedModels/{tuned_name}-{id}",
                        id=f"{tuned_name}-{id+1}",
                        epoch_count=5,
                        batch_size=2,
                        learning_rate=0.001,
                        training_data=extracted_data,
                        )
                        id = id + 1 # increment this
                        print(f"generated {tuned_name}-{id}")

                        for status in operation.wait_bar():
                            time.sleep(10)

                result = operation.result()
                print(result)
                model_name = result.name
                
                # Plot and save the loss curve
                snapshots = pd.DataFrame(result.tuning_task.snapshots)
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=snapshots, x='epoch', y='mean_loss')
                plt.title('Loss Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Mean Loss')
                plt.savefig(os.path.join(output_directory, 'loss_curve.png'))
                plt.close()

                # # You can plot the loss curve with:
                # snapshots = pd.DataFrame(result.tuning_task.snapshots)
                # sns.lineplot(data=snapshots, x='epoch', y='mean_loss')

def main(input_directory, output_directory):
    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of folders containing JSON files.")
    parser.add_argument("--input_directory", type=str, default="filtered_dataset/gemini-1.5-flash-8b/MATH/train", help="Path to the directory containing folders with JSON files.")
    parser.add_argument("--output_directory", type=str, default=f"./training/{base_name}/{id}", help="Path to the directory to save figure.")
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
