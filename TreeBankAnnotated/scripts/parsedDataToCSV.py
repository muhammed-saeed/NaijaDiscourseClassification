import json
import ast

# Read the JSON file
with open('/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/parsedDataDiscopy/TreeBankRealEnglishAnnotationTest.json') as f:
    data = json.load(f)

texts = []
args1 = []
args2 = []
connectives = []
relationss = []

typeRelation = []
for sentence_dict in data:
    # Extract the English Annotated Sentences Parsed
    annotated_sentence = sentence_dict["English Annotated Sentences Parsed"]

    # Parse the annotated sentence string into a dictionary
    parsed_dict = ast.literal_eval(annotated_sentence)

    # Access the desired information from the parsed dictionary
    text = parsed_dict['text']
    relations = parsed_dict['relations']
    texts.append(text)

    for relation in relations:
        arg1 = relation['Arg1']['RawText']
        arg2 = relation['Arg2']['RawText']
        connective = relation['Connective']['RawText']
        args1.append(arg1)
        args2.append(arg2)
        connectives.append(connective)
        relationss.append(relation)
        typeRelation.append(relation['Type'])

print("Number of texts:", len(texts))
print("Number of relations:", len(relationss))
print("First relation:", relationss[0])
print("Number of args1:", len(args1))
print("Number of args2:", len(args2))
print("Number of connectives:", len(connectives))
print(len(typeRelation))
# print(relationss[0])