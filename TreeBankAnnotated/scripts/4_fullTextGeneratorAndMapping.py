import argparse
import pandas as pd
import numpy as np

def main(args):
    annotated_df = pd.read_csv(args.csv_file_explicit_implicit)
    treebank_df = pd.read_csv(args.treebank_df)

    # Initialize list to store full texts
    full_texts = []

    # Iterate over rows in annotated_df
    for _, row in annotated_df.iterrows():
        arg1_span = str(row['arg1span'])
        arg2_span = str(row['arg2span'])
        connective_span = str(row['connspan'])

        # Check if connective_span is NaN
        if pd.isna(connective_span):
            if arg1_span < arg2_span:
                full_text = f"{row['arg1raw']} {row['arg2raw']}"
            else:
                full_text = f"{row['arg2raw']} {row['arg1raw']}"
        elif ';' in arg1_span or ';' in arg2_span or ';' in connective_span:
            full_text = "neglect for now"
        else:
            # Process connective_span
            if connective_span.lower() == 'nan':
                if arg1_span.lower() !='nan' and arg2_span.lower() !='nan' :
                    full_text = row['arg1raw'] + row['arg2raw'] 
                    # full_text = row['arg1raw']
                elif arg1_span.lower() == 'nan':
                    full_text = row['arg2raw']
                elif arg2_span.lower() == 'nan':
                    full_text = row['arg1raw']
            else:
                try:
                    # Convert span values to integers
                    arg1_start, arg1_end = map(int, arg1_span.split('..'))
                    arg2_start, arg2_end = map(int, arg2_span.split('..'))
                    connective_start, connective_end = map(int, connective_span.split('..'))

                    # Check if arg1span < connspan < arg2span
                    if arg1_start < connective_start < arg2_start:
                        full_text = f"{row['arg1raw']} {row['conn_raw']} {row['arg2raw']}"
                    elif arg2_start < connective_start < arg1_start:
                        full_text = f"{row['arg2raw']} {row['conn_raw']} {row['arg1raw']}"
                    elif connective_start < arg1_start < arg2_start:
                        full_text = f"{row['conn_raw']} {row['arg1raw']} {row['arg2raw']}"
                    else:
                        full_text = f"{row['conn_raw']} {row['arg2raw']} {row['arg1raw']}"
                except ValueError:
                    full_text = "skip for now"

        full_texts.append(full_text)

    # Add full_text column to annotated_df
    annotated_df['full_text'] = full_texts

    print(annotated_df.head())
    annotated_df.to_csv(args.fullText, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_explicit_implicit", default="TreeBankAnnotated/csv/treebank_nur_implicit_explicit.csv")
    parser.add_argument('--treebank_df', default="TreeBankAnnotated/csv/conllu/allDataFramesConllu.csv")
    parser.add_argument("--fullText", default="TreeBankAnnotated/csv/processed/mergedAnnotationFullText.csv")
    args = parser.parse_args()
    main(args)
