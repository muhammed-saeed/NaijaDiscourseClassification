import argparse
import pandas as pd

def main(args):
    df = pd.read_csv(args.csv_file, low_memory=False)
    print(df['reltype'].value_counts())
    filtered_df = df[df['reltype'].isin(['Explicit', 'Implicit'])]

    # Print the resulting filtered DataFrame
    print(filtered_df['reltype'].value_counts())
    filtered_df.to_csv(args.csv_file_explicit_implicit)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="The Dataset ")
    parser.add_argument("--csv_file", type = str, default="TreeBankAnnotated/merged_annotations_added_with_uttid_clean.csv")
    parser.add_argument("--csv_file_explicit_implicit", type=str, default="TreeBankAnnotated/csv/treebank_nur_implicit_explicit.csv")
    parser.add_argument('--fulltext', type=str)
    args = parser.parse_args()
    main(args)
