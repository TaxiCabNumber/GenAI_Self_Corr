# GenAI_Self_Corr
10-423/623 Generative AI Project on LLM Self Correction for Quantitative Tasks

# Code structure
At a high level, the repository 
(1) generates a dataset by concatenating multiple inferences
(2) finetunes a LLM with the made dataset
(3) compares the base model and the fine-tuned model on the test split to see if self-correction was learned

For (1) the files involved are
- utils.py
- generate_data.py
- prompts.py
- key.py
The Box folder has the datasets for training (GSM8K, codecontests) and validation (MATH, HumanEval)

For (2) the files involved are 
- STaR_pair_train.py
- GSM8K_check_self_corr.py
- MATH_check_self_corr.py
- prompts.py
- key.py

Batch/shell scripts were written to execute the Python calls on an automated routine to make pipelines (1) and (2) into an executable 

The Jupyter notebook runs a naive example that can be replaced with GSM8K. The second file takes the output of generate_data.py as the training data to fine-tune the model to generate responses in the same fashion as the training data

The most basic implementation isolates a single numerical value from the inference and sees if it matches the value in the ground truth. 

However, tHe current implementation is to only count the actual tokens after "Answer" as seen in "... Final Answer:", which is in accordance to the format of inferences solicited by the prompts.

The following two "..._train.py" files base their methods off STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning (Zelikman et.al. 2022) and SFT by directly feeding examples of two rounds of self-correction.

The most basic way of replicating the baseline is to use pair-wise supervised fine-tuning in the Google genai API
This is done by generaing quality samples from the GS8M8K or the small size TheVault dataset
