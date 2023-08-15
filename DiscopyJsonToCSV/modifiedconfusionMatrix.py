import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the csv files into pandas dataframes
real_df = pd.read_csv('/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/RealEnglishDiscopyPrased.csv')
translated_df = pd.read_csv('/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/translatedEnglishDiscopyPrased.csv')

# Count occurrences of each file_index in both DataFrames
real_counts = real_df['file_index'].value_counts()
translated_counts = translated_df['file_index'].value_counts()

# Align the two Series to have the same set of indices (file_index values)
aligned_real_counts, aligned_translated_counts = real_counts.align(translated_counts, fill_value=0)

# Now, identify file-indices with the same count in both aligned Series
matching_indices = aligned_real_counts[aligned_real_counts == aligned_translated_counts].index

# Filter rows based on matching file-indices
real_df = real_df[real_df['file_index'].isin(matching_indices)]
translated_df = translated_df[translated_df['file_index'].isin(matching_indices)]


import pandas as pd
from sklearn.metrics import confusion_matrix

# ... [loading data and preprocessing as per your existing code]

def generate_confusion_matrix_for_column(real_df, translated_df, column, file):
    labels = sorted(list(set(real_df[column])))  # All unique labels in real_df for that column

    matrices = {}
    
    for value in labels:
        # Get indices of rows in real_df where the column has the specific value
        indices = real_df[real_df[column] == value].index
        
        # Ensure we are only considering indices that exist in both dataframes
        common_indices = indices[indices.isin(translated_df.index)]
        
        true_values = real_df.loc[common_indices, column].tolist()
        pred_values = translated_df.loc[common_indices, column].tolist()

        matrix = confusion_matrix(true_values, pred_values, labels=labels)
        
        # Extracting the specific row of interest from the matrix
        matrices[value] = matrix[labels.index(value)]

    # Converting to DataFrame for clarity
    df_matrix = pd.DataFrame(matrices, index=labels, columns=labels).T
    
    file.write(f"Confusion Matrix for {column}:\n")
    file.write(df_matrix.to_string())
    file.write("\n" + "-" * 50 + "\n")


# ... [replacement of NaNs as per your existing code]
# Replace NaNs
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

# # Analyze each column
# for column in ['Connective_RawText', 'Sense', 'Type']:
#     generate_confusion_matrix_for_column(real_df, translated_df, column)
# Before the for loop that analyzes each column
file_paths = {
    'Connective_RawText': '/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/connective_output.txt',
    'Sense': '/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/sense_output.txt',
    'Type': '/local/musaeed/NaijaDiscourseClassification/DiscopyJsonToCSV/results/type_output.txt'
}

# Inside the for loop
for column in ['Connective_RawText', 'Sense', 'Type']:
    with open(file_paths[column], 'w') as file:
        generate_confusion_matrix_for_column(real_df, translated_df, column, file)
