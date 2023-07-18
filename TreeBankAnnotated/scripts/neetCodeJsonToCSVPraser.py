import pandas as pd
import json
import ast
import numpy as np

# Read the CSV file into a dataframe
df = pd.read_csv('/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/processed/mergedTextWithPCMFullTextAndTranslated.csv')

def extract_data_from_json(json_file_path, code):
    # Read the JSON file
    with open(json_file_path) as f:
        data = json.load(f)

    texts = []
    args1 = []
    args2 = []
    connectives = []
    relationss = []
    typeRelation = []

    unique_relations = set()  # Keep track of unique relations

    for sentence_dict in data:
        annotated_sentence = sentence_dict[code]
        parsed_dict = ast.literal_eval(annotated_sentence)

        text = parsed_dict['text']
        relations = parsed_dict['relations']
        texts.append(text)

        for relation in relations:
            arg1 = relation['Arg1']['RawText']
            arg2 = relation['Arg2']['RawText']
            connective = relation['Connective']['RawText']
            Type = relation['Type']
            # Replace empty strings with np.nan
            arg1 = arg1 if arg1 != '' else np.nan
            arg2 = arg2 if arg2 != '' else np.nan
            connective = connective if connective != '' else np.nan
            Type = Type if Type != '' else np.nan

            # Check if relation is unique
            relation_key = (arg1, arg2, connective, Type)
            if relation_key not in unique_relations:
                args1.append(arg1)
                args2.append(arg2)
                connectives.append(connective)
                relationss.append(relation)
                typeRelation.append(Type)
                unique_relations.add(relation_key)

    print("Number of texts:", len(texts))
    print("Number of relations in JSON file:", len(data))
    print("Number of unique relations generated:", len(relationss))
    print("Number of args1:", len(args1))
    print("Number of args2:", len(args2))
    print("Number of connectives:", len(connectives))

    return texts, args1, args2, connectives, relationss, typeRelation

en_texts, en_args1, en_args2, en_connectives, en_relationss, enTypeRelation = extract_data_from_json("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/parsedDataDiscopy/TreeBankRealEnglishAnnotationTest.json", code="English Annotated Sentences Parsed")

pcm_texts, pcm_args1, pcm_args2, pcm_connectives, pcm_relationss, pcmTypeRelation = extract_data_from_json("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/parsedDataDiscopy/translatedPCMParseFileTest.json", code="translatedDetails")

data = {
    'en_texts': en_texts,
    'en_args1': en_args1,
    'en_args2': en_args2,
    'en_connectives': en_connectives,
    'en_relationss': en_relationss,
    'enTypeRelation': enTypeRelation,
    'pcm_texts': pcm_texts[:len(en_texts)],
    'pcm_args1': pcm_args1[:len(en_texts)],
    'pcm_args2': pcm_args2[:len(en_texts)],
    'pcm_connectives': pcm_connectives[:len(en_texts)],
    'pcm_relationss': pcm_relationss[:len(en_texts)],
    'pcmTypeRelation': pcmTypeRelation[:len(en_texts)]
}

df = pd.DataFrame(data)
