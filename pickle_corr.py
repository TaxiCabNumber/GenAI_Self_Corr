import json
import os
import pickle
import argparse
from utils import dataloader, load_one_data, generate_inference, save_inferences_to_json, compile_training_dataset
from prompts import GSM8K_initial_prompt, GSM8K_self_correcting_prompt

def save_idx(output_path, idx):
    with open(os.path.join(output_path, "current_idx.pkl"), "wb") as f:
        pickle.dump(idx, f)




