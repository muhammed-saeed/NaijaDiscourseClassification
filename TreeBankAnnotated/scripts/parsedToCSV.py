import json
import pandas as pd

# Read JSON data from file
with open('data.json') as file:
    data = json.load(file)

# Initialize lists to store the extracted information
arg1_list = []
arg2_list = []
connective_list = []
relation_type_list = []

# Process each item in the data
for item in data:
    parsed_data = item["English Annotated Sentences Parsed"]
    text = parsed_data["text"]
    relations = parsed_data.get("relations", [])
    
    if relations:
        for relation in relations:
            arg1 = relation.get("arg1")
            arg2 = relation.get("arg2")
            connective = relation.get("connective")
            relation_type = relation.get("type", "implicit")
            
            arg1_list.append(arg1)
            arg2_list.append(arg2)
            connective_list.append(connective)
            relation_type_list.append(relation_type)
    else:
        arg1_list.append(None)
        arg2_list.append(None)
        connective_list.append(None)
        relation_type_list.append("implicit")

# Create a DataFrame from the extracted information
df = pd.DataFrame({
    "arg1": arg1_list,
    "arg2": arg2_list,
    "connective": connective_list,
    "relation_type": relation_type_list
})

# Print the DataFrame
print(df)
