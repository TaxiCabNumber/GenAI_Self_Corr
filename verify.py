# # import re

# # def extract_number_after_pounds(response):
# #     """
# #     Extracts the number after 'I hope it is correct' or '####' from the response string.

# #     Args:
# #         response (str): The response string.

# #     Returns:
# #         str: The extracted number or an empty string if no match is found.
# #     """
# #     last_correct_index = response.rfind('I hope it is correct')
# #     last_pounds_index = response.rfind('####')

# #     if last_correct_index != -1 and (last_correct_index > last_pounds_index or last_pounds_index == -1):
# #         split_part = response[last_correct_index + len('I hope it is correct'):]
# #     elif last_pounds_index != -1:
# #         split_part = response[last_pounds_index + len('####'):]
# #     else:
# #         return ""

# #     # Extract numbers from the split part
# #     match = re.search(r'(\$?\d+)', split_part)
# #     if match:
# #         return match.group(1)
# #     return ""

# # parse = "1. Clips sold in April: 48\n2. Clips sold in May: 48 / 2 = 24\n3. Total clips sold: 48 + 24 = 72\n\nI hope it is correct: 72 $clips$\n"
# # ground_truth = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"

# # print(extract_number_after_pounds(parse)) # 72
# # # print(extract_number_after_pounds(ground_truth)) # 72

# import json
# import os

# def extract_content_from_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#         extracted_data = []
#         for entry in data:
#             if "question" in entry and "inference" in entry:
#                 extracted_data.append({
#                     "text_input": entry["question"],
#                     "output": entry["inference"]
#                 })
#         return extracted_data

# def process_directory(input_directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     for folder_name in os.listdir(input_directory):
#         folder_path = os.path.join(input_directory, folder_name)
#         if os.path.isdir(folder_path):
#             all_extracted_data = []
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith(".json"):
#                     file_path = os.path.join(folder_path, file_name)
#                     extracted_data = extract_content_from_json(file_path)
#                     print(extracted_data)
#                     all_extracted_data.extend(extracted_data)
#                     print(all_extracted_data)

# process_directory("./filtered_dataset/gemini-1.5-flash-8b/MATH/train/", "check")

import os
import json
import argparse
import google.generativeai as genai
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


base_name = "models/gemini-1.5-flash-8b-001-tuning"
# models/gemini-1.0-pro-001
# models/gemini-1.5-flash-001-tuning

id = "math-train-001"

from key import google_api_key 
genai.configure(api_key=google_api_key) # api_key = userdata.get('GOOGLE_API_KEY')
initialize_model = True

for i, m in zip(range(5), genai.list_tuned_models()):
  print(m.name)

# print("pop \n")

base_model = [
    m for m in genai.list_models()
    if "createTunedModel" in m.supported_generation_methods and
    "pro" in m.name][0]
print(base_model)