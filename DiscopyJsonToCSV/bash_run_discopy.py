import subprocess
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

df = pd.read_csv(
    "/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/csv/devTestConlluDatasetwithTranslationsOfPCM.csv")
english_real_annotation = df['EnglishTranslationPCMWithoutDEVTest'].tolist()

# Loop through the list with tqdm progress bar
for idx, string in tqdm(enumerate(english_real_annotation), total=len(english_real_annotation), desc="Processing"):
    # Save the string to a text file
    input_file_path = f"/local/musaeed/discopy/input_{idx}.txt"
    with open(input_file_path, "w") as input_file:
        input_file.write(string)

    # Construct the command
    tokenize_command = f"discopy-tokenize -i {input_file_path}"
    add_parses_command = "discopy-add-parses -c "
    nn_parse_command = f"discopy-nn-parse bert-base-cased /local/musaeed/discopy/models/lin/pipeline-bert-2/ > /local/musaeed/discopy/fake/{idx}.json"

    # Combine the commands using pipes
    combined_command = f"{tokenize_command} | {add_parses_command} | {nn_parse_command}"

    # Run the combined command in the terminal
    subprocess.run(combined_command, shell=True)

    # Remove the input file after processing
    subprocess.run(["rm", input_file_path])
