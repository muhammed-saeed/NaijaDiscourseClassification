import json
import pandas as pd
import numpy as np

# Read JSON data from file
with open('/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/parsedDataDiscopy/RealEnglishFilteredTreeBankPCMDevTest.json', 'r') as file:
    data = json.load(file)

# Extract required information
texts = []
arg1_rawtexts = []
arg2_rawtexts = []
connectives = []
relation_types = []
senses = []

for item in data:
    item_data = item["English Annotated Sentences Parsed"]
    # item_data = item['translatedDetails']
    text = item_data["text"]
    texts.append(text if text else np.nan)
    relations = item_data["relations"]
    if relations:
        relation = relations[0]  # Assuming there is only one relation per item
        arg1_rawtext = relation["Arg1"]["RawText"]
        arg1_rawtexts.append(arg1_rawtext if arg1_rawtext else np.nan)
        arg2_rawtext = relation["Arg2"]["RawText"]
        arg2_rawtexts.append(arg2_rawtext if arg2_rawtext else np.nan)
        connective = relation["Connective"]["RawText"]
        connectives.append(connective if connective else np.nan)
        relation_types.append(relation["Type"])
        sense = relation["Sense"]
        sense = sense[0] if sense else np.nan
        senses.append(sense)
    else:
        arg1_rawtexts.append(np.nan)
        arg2_rawtexts.append(np.nan)
        connectives.append(np.nan)
        relation_types.append(np.nan)
        senses.append(np.nan)

# Create a DataFrame
df = pd.DataFrame({
    "Text": texts,
    "Arg1 RawText": arg1_rawtexts,
    "Arg2 RawText": arg2_rawtexts,
    "Connective": connectives,
    "Relation Type": relation_types,
    "Sense": senses
})

# Save the DataFrame to a file
df.to_csv('/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/ParsedCSV/TreebankRealParsedToCsv.csv', index=False)
