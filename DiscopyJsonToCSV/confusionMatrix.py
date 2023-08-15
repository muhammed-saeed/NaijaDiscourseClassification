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

def generate_confusion_matrix_for_column(real_df, translated_df, column):
    unique_values = real_df[column].unique()

    for value in unique_values:
        print(f"Analyzing {column} value: {value}")

        # Get indices of rows in real_df where the column has the specific value
        indices = real_df[real_df[column] == value].index
        
        # Ensure we are only considering indices that exist in both dataframes
        common_indices = indices[indices.isin(translated_df.index)]

        true_values = real_df.loc[common_indices, column].tolist()
        pred_values = translated_df.loc[common_indices, column].tolist()

        labels = list(set(true_values) | set(pred_values))
        
        matrix = confusion_matrix(true_values, pred_values, labels=labels)
        
        # Printing the confusion matrix
        print(f"Confusion Matrix for {column} value: {value}")
        print(matrix)
        print("-" * 50)

# Replace NaNs
real_df = real_df.fillna({
    'Type': 'Implicit',
    'Sense': 'NoSense',
    'Connective_RawText': 'NoConnective'
})
translated_df = translated_df.fillna({
    'Type': 'Implicit',
    'Sense': 'NoSense',
    'Connective_RawText': 'NoConnective'
})

# Analyze each column
for column in ['Connective_RawText', 'Sense', 'Type']:
    generate_confusion_matrix_for_column(real_df, translated_df, column)
