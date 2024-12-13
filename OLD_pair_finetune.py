import google.generativeai as genai
import pandas as pd
import time
import os

# Function to save the model ID to a file


# Define the base model and training data
base_model = "models/gemini-1.5-flash-001-tuning"
training_data = [
    {"text_input": "1", "output": "2"},
    {"text_input": "seven", "output": "eight"},
    # Add more training data as needed
]

# Define the model ID file path
model_id_file = 'tuned_model_id.txt'

# Load the model ID if it exists
model_id = load_model_id(model_id_file)

if model_id and model_exists(model_id):
    print(f"Using existing tuned model: {model_id}")
    source_model = f'tunedModels/{model_id}'
else:
    # Create a new tuned model
    model_id = f'generate-num-{random.randint(0, 10000)}'
    source_model = base_model

operation = genai.create_tuned_model(
    display_name="increment",
    source_model=source_model,
    training_data=training_data,
    id=model_id,
    epoch_count=20,
    batch_size=4,
    learning_rate=0.001,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()
print(result)


# You can plot the loss curve with:
# snapshots = pd.DataFrame(result.tuning_task.snapshots)
# sns.lineplot(data=snapshots, x='epoch', y='mean_loss')

model = genai.GenerativeModel(model_name=result.name)

# Example usage of the model
result = model.generate_content("III")
print(result.text)  # IV