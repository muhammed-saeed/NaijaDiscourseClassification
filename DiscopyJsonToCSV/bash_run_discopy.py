import subprocess
import pandas as pd
from tqdm import tqdm
import argparse

def main(args):
    df = pd.read_csv(args.csv_path)
    english_real_annotation = df[args.column_name].tolist()

    for idx, string in tqdm(enumerate(english_real_annotation), total=len(english_real_annotation), desc="Processing"):
        input_file_path = f"{args.output_folder}/input_{idx}.txt"
        with open(input_file_path, "w") as input_file:
            input_file.write(string)

        tokenize_command = f"discopy-tokenize -i {input_file_path}"
        add_parses_command = "discopy-add-parses -c "
        nn_parse_command = f"discopy-nn-parse {args.parse_model} {args.nn_model} > {args.output_folder}/{idx}.json"

        combined_command = f"{tokenize_command} | {add_parses_command} | {nn_parse_command}"

        subprocess.run(combined_command, shell=True)

        subprocess.run(["rm", input_file_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text using discopy tools.")
    parser.add_argument("--csv_path", type=str, default="/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/csv/devTestConlluDatasetwithTranslationsOfPCM.csv", help="Path to the CSV file")
    parser.add_argument("--column_name", type=str, default="EnglishTranslationPCMWithoutDEVTest", help="Name of the column containing text")
    parser.add_argument("--output_folder", type=str, default="/local/musaeed/discopy/fake", help="Path to the output folder")
    parser.add_argument("--parse_model", type=str, default="bert-base-cased", help="Parse model name")
    parser.add_argument("--nn_model", type=str, default="/local/musaeed/discopy/models/lin/pipeline-bert-2/", help="Path to the NN model")

    args = parser.parse_args()
    main(args)
