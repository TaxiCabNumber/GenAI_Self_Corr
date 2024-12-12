import os
import matplotlib.pyplot as plt

def collect_data(directory):
    folder_paths = []
    first_accuracies = []
    second_accuracies = []

    for file_name in os.listdir(directory):
        if file_name.endswith("_acc_sum.txt"):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                folder_path = lines[0].split(": ")[1].strip().split("/")[-1]
                first_accuracy = float(lines[1].split(": ")[1].strip())
                second_accuracy = float(lines[2].split(": ")[1].strip())

                folder_paths.append(folder_path)
                first_accuracies.append(first_accuracy)
                second_accuracies.append(second_accuracy)

    return folder_paths, first_accuracies, second_accuracies

def plot_accuracies(folder_paths, first_accuracies, second_accuracies):
    x = range(len(folder_paths))

    fig, ax = plt.subplots()
    bar_width = 0.35

    bar1 = plt.bar(x, first_accuracies, bar_width, label='First Accuracy')
    bar2 = plt.bar([p + bar_width for p in x], second_accuracies, bar_width, label='Second Accuracy')

    plt.xlabel('Folder Path')
    plt.ylabel('Accuracy')
    plt.title('First and Second Accuracies by Folder Path')
    plt.xticks([p + bar_width / 2 for p in x], folder_paths, rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = "./Gemini_base_sc_results"  # Change this to your directory path
    folder_paths, first_accuracies, second_accuracies = collect_data(directory)
    plot_accuracies(folder_paths, first_accuracies, second_accuracies)