import pandas as pd
from sklearn.metrics import confusion_matrix
import argparse

def generate_confusion_matrix_for_column(real_df, translated_df, column, file):
    labels = sorted(list(set(real_df[column])))

    matrices = {}
    
    for value in labels:
        indices = real_df[real_df[column] == value].index
        common_indices = indices[indices.isin(translated_df.index)]
        
        true_values = real_df.loc[common_indices, column].tolist()
        pred_values = translated_df.loc[common_indices, column].tolist()

        matrix = confusion_matrix(true_values, pred_values, labels=labels)
        matrices[value] = matrix[labels.index(value)]

    df_matrix = pd.DataFrame(matrices, index=labels, columns=labels).T
    
    file.write(f"Confusion Matrix for {column}:\n")
    file.write(df_matrix.to_string())
    file.write("\n" + "-" * 50 + "\n")
    
    csv_path = f"/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/{column}_confusion_matrix.csv"
    df_matrix.to_csv(csv_path)

def main(args):
    real_df = pd.read_csv(args.real_csv_path)
    translated_df = pd.read_csv(args.translated_csv_path)
    
    real_counts = real_df['file_index'].value_counts()
    translated_counts = translated_df['file_index'].value_counts()
    
    aligned_real_counts, aligned_translated_counts = real_counts.align(translated_counts, fill_value=0)
    
    matching_indices = aligned_real_counts[aligned_real_counts == aligned_translated_counts].index
    
    real_df = real_df[real_df['file_index'].isin(matching_indices)]
    translated_df = translated_df[translated_df['file_index'].isin(matching_indices)]
    
    real_df = real_df.fillna({
        'Type': 'Missing',
        'Sense': 'NoSense',
        'Connective_RawText': 'NoConnective'
    })
    translated_df = translated_df.fillna({
        'Type': 'Missing',
        'Sense': 'NoSense',
        'Connective_RawText': 'NoConnective'
    })
    
    file_paths = {
        'Connective_RawText': '/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/connective_output.txt',
        'Sense': '/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/sense_output.txt',
        'Type': '/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/type_output.txt'
    }

    for column in ['Connective_RawText', 'Sense', 'Type']:
        with open(file_paths[column], 'w') as file:
            generate_confusion_matrix_for_column(real_df, translated_df, column, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and generate confusion matrices.")
    parser.add_argument("--real_csv_path", type=str, default="/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/RealEnglishDiscopyPrased.csv", help="Path to the real CSV file")
    parser.add_argument("--translated_csv_path", type=str, default="/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/translatedEnglishDiscopyPrased.csv", help="Path to the translated CSV file")

    args = parser.parse_args()
    main(args)
