import pandas as pd

# Assuming you have the original dataframe called 'original_df' and the checklist as a Python list of sentences called 'checklist'

original_df = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/processed/mergedTextWithPCMFullText.csv")

# Sample checklist
checklist_df = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/EnglishTreeBankDevTestData.csv")
checklist = checklist_df['check_list1'].tolist()

# Create a new DataFrame (devTest) containing rows where checklist sentences are part of arg1, arg2, or fulltext
devTest = original_df[
    (original_df['arg1raw'].apply(lambda x: isinstance(x, str) and any(sentence in x for sentence in checklist))) |
    (original_df['arg2raw'].apply(lambda x: isinstance(x, str) and any(sentence in x for sentence in checklist))) |
    (original_df['full_text'].apply(lambda x: isinstance(x, str) and any(sentence in x for sentence in checklist)))
]

# Reset the index of the new DataFrame
devTest.reset_index(drop=True, inplace=True)

# Now, devTest contains rows where any of the sentences in the checklist are present in arg1, arg2, or fulltext.
print(len(devTest))
devTest.to_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/csv/devTestdataset.csv", index=False)