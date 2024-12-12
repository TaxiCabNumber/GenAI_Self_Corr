@echo off
REM Batch file to run the generate_dataset.py script with configuration parameters in an Anaconda environment

REM Enable delayed variable expansion
setlocal enabledelayedexpansion

REM Set the number of API call cycles to run to go through enough data
set N=2
set SUBJECTS=0

REM Set other parameters
set MODEL_NAME=gemini-1.5-flash-8b
set NUMBER_OF_ROUNDS=1
set INIT_PROMPT="You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form 'I hope it is correct: #### $answer$'"
set SELF_CORR_PROMPT="There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form 'I hope it is correct: #### $answer$'"
set DATASET_FORMAT=jsonl

REM Define the arrays of dataset paths and output paths
set DATASET_PATHS[0]=./data/GSM8K_test.jsonl

set OUTPUT_PATHS[0]=./generated_chains/%MODEL_NAME%/GSM8K/test


REM Activate the Anaconda environment
call conda activate mingpt

REM Navigate to the directory containing the Python scripts
cd /d C:\Users\joncc\Documents\GitHub\GenAI_Self_Corr

REM Loop over the number of iterations
for /L %%i in (1,1,%N%) do (
    REM Loop over the subjects
    for %%j in (%SUBJECTS%) do (
        REM Get the corresponding dataset path and output path by index
        set DATASET_PATH=!DATASET_PATHS[%%j]!
        set OUTPUT_PATH=!OUTPUT_PATHS[%%j]!

        REM Run the Python script with the parameters
        python GSM8K_generate_data.py --dataset_path !DATASET_PATH! --model_name %MODEL_NAME% --output_path !OUTPUT_PATH! --number_of_rounds %NUMBER_OF_ROUNDS% --init_prompt %INIT_PROMPT% --self_corr_prompt %SELF_CORR_PROMPT% --dataset_format %DATASET_FORMAT%
            
        REM Pause for 20 seconds to reset the API call limit
        timeout /t 20 /nobreak
    )
)

REM Loop over output paths to generate plots
for %%j in (%SUBJECTS%) do (
    REM Get the corresponding dataset path and output path by index
    set OUTPUT_PATH=!OUTPUT_PATHS[%%j]!

    REM Run the Python script with the parameters
    python GSM8K_check_self_corr.py --folder_path !OUTPUT_PATH! --model_name %MODEL_NAME% --mode "test"
    
    REM Pause for 1 second to allow the file to be saved
    timeout /t 1 /nobreak
)

REM Deactivate the Anaconda environment
call conda deactivate

pause