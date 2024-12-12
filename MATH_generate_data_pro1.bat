@echo off
REM Batch file to run the generate_dataset.py script with configuration parameters in an Anaconda environment

REM Enable delayed variable expansion
setlocal enabledelayedexpansion

REM Set the number of API call cycles to run to go through enough data
REM Leave "N=" undefined to skip generation and only evaluate
set N=
set SUBJECTS=0 1 2 3 4 5 6

REM Define the arrays of dataset paths and output paths
set DATASET_PATHS[0]=./data/MATH/train/counting_and_probability
set DATASET_PATHS[1]=./data/MATH/train/number_theory
set DATASET_PATHS[2]=./data/MATH/train/prealgebra
set DATASET_PATHS[3]=./data/MATH/train/algebra
set DATASET_PATHS[4]=./data/MATH/train/intermediate_algebra
set DATASET_PATHS[5]=./data/MATH/train/precalculus
set DATASET_PATHS[6]=./data/MATH/train/geometry

REM Set other parameters
set MODEL_NAME=gemini-1.0-pro
set NUMBER_OF_ROUNDS=1
set INIT_PROMPT="You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form 'Final Answer: \\boxed{$answer$}. I hope it is correct.'"
set SELF_CORR_PROMPT="There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form 'Final Answer: \\boxed{$answer$}. I hope it is correct.'"
set DATASET_FORMAT=json

set OUTPUT_PATHS[0]=./generated_chains/%MODEL_NAME%/MATH/train/counting_and_probability
set OUTPUT_PATHS[1]=./generated_chains/%MODEL_NAME%/MATH/train/number_theory
set OUTPUT_PATHS[2]=./generated_chains/%MODEL_NAME%/MATH/train/prealgebra
set OUTPUT_PATHS[3]=./generated_chains/%MODEL_NAME%/MATH/train/algebra
set OUTPUT_PATHS[4]=./generated_chains/%MODEL_NAME%/MATH/train/intermediate_algebra
set OUTPUT_PATHS[5]=./generated_chains/%MODEL_NAME%/MATH/train/precalculus
set OUTPUT_PATHS[6]=./generated_chains/%MODEL_NAME%/MATH/train/geometry

REM Activate the Anaconda environment
REM conda activate mingpt

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
        python MATH_generate_data.py --dataset_path !DATASET_PATH! --model_name %MODEL_NAME% --output_path !OUTPUT_PATH! --number_of_rounds %NUMBER_OF_ROUNDS% --init_prompt %INIT_PROMPT% --self_corr_prompt %SELF_CORR_PROMPT% --dataset_format %DATASET_FORMAT%
            
        REM Pause for 20 seconds to reset the API call limit
        timeout /t 20 /nobreak
    )
)

REM Loop over output paths to generate plots
for %%j in (%SUBJECTS%) do (
    REM Get the corresponding dataset path and output path by index
    set OUTPUT_PATH=!OUTPUT_PATHS[%%j]!

    REM Run the Python script with the parameters
    python MATH_check_self_corr.py --folder_path !OUTPUT_PATH! --model_name %MODEL_NAME% --mode "train"
    
    REM Pause for 1 second to allow the file to be saved
    timeout /t 1 /nobreak
)

REM Deactivate the Anaconda environment
REM conda deactivate

pause