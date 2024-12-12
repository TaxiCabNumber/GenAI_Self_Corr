import os
import shutil
import argparse

'''
Usage:
Use the following command to run the script, specifying the path to the directory containing the input text files:
This setup allows you to process all text files in the specified directory, checking for matching files and copying them to the new directory with the modified path.

python curate_dataset.py path/to/input_directory
'''

def parse_text_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.split(": ", 1)
            if key == "Improvement Indices":
                data[key] = [int(x) for x in value.strip().strip('[]').split(', ')]
            else:
                data[key] = value.strip()
    return data

def check_directory_for_files(folder_path, indices):
    matching_files = []
    for file_name in os.listdir(folder_path):
        for index in indices:
            if file_name.endswith(f"_{index}.json"):
                matching_files.append(file_name)
    return matching_files

def copy_files_to_new_directory(folder_path, matching_files, new_root):
    new_folder_path = folder_path.replace(folder_path.split('/')[1], new_root)
    os.makedirs(new_folder_path, exist_ok=True)
    
    for file_name in matching_files:
        src_file = os.path.join(folder_path, file_name)
        dest_file = os.path.join(new_folder_path, file_name)
        shutil.copy(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")

def main(input_directory):
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".txt"):
            input_file = os.path.join(input_directory, file_name)
            data = parse_text_file(input_file)
            
            folder_path = data.get("Folder Path")
            improvement_indices = data.get("Improvement Indices", [])
            
            if folder_path and improvement_indices:
                matching_files = check_directory_for_files(folder_path, improvement_indices)
                new_root = "filtered_dataset"
                copy_files_to_new_directory(folder_path, matching_files, new_root)
            else:
                print(f"Folder Path or Improvement Indices not found in the input file: {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of input files.")
    parser.add_argument("input_directory", type=str, nargs='?', default="./results/gemini-1.5-flash-8b/MATH/train", help="Path to the directory containing input text files.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_directory):
        print(f"Error: The directory {args.input_directory} does not exist.")
        exit(1)
    
    main(args.input_directory)