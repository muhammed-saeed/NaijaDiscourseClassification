import pandas as pd
import os
import logging
import sacrebleu
import pandas as pd
import random 
import argparse
from simpletransformers.t5 import T5Model, T5Args
import torch

torch.manual_seed(0)
random.seed(0)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 128
model_args.length_penalty = 1
model_args.num_beams = 5
model_args.eval_batch_size = 32


model_path = "/local/musaeed/BESTT5TranslationModel"

model = T5Model("t5", model_path, args=model_args)
def search_treebank(treebank, full_text, arg1raw, arg2raw, connraw):
    pcm_full_text = []

    for i in range(len(full_text)):
        # Get the values at index i from each series
        f_text = full_text[i]
        a1_raw = arg1raw[i]
        a2_raw = arg2raw[i]
        c_raw = connraw[i]

        # Check if conn_raw is NaN and replace it with an empty string
        if pd.isna(c_raw):
            c_raw = ""

        # Check if arg2raw is NaN and replace it with an empty string
        if pd.isna(a2_raw):
            a2_raw = ""

        # Check if arg1raw is NaN and replace it with an empty string
        if pd.isna(a1_raw):
            a1_raw = ""

        # Step 1: Search for full text in treebank['Text_en']
        if f_text == "neglect for now":
            pcm_full_text.append("neglect for now")

        else:
            matching_rows = treebank.loc[treebank['Text_en'].str.contains(f_text, case=False, na=False, regex=False)]
            if not matching_rows.empty:
                corresponding_value_pcm = matching_rows['Text_ortho'].values[0]
                pcm_full_text.append(corresponding_value_pcm)
            else:
                # Step 2: Search for arg1 in treebank['Text_en']
                matching_rows = treebank.loc[treebank['Text_en'].str.contains(a1_raw, case=False, na=False, regex=False)]
                if not matching_rows.empty:
                    corresponding_value_pcm1 = matching_rows['Text_ortho'].values[0]

                    # Step 3: Search for conn1 + arg2 in treebank['Text_en']
                    conn_arg2_text = c_raw + a2_raw
                    matching_rows = treebank.loc[treebank['Text_en'].str.contains(conn_arg2_text, case=False, na=False, regex=False)]
                    if not matching_rows.empty:
                        corresponding_value_pcm2 = matching_rows['Text_ortho'].values[0]
                    else:
                        # Step 4: Search for arg2 in treebank['Text_en']
                        matching_rows = treebank.loc[treebank['Text_en'].str.contains(a2_raw, case=False, na=False, regex=False)]
                        if not matching_rows.empty:
                            corresponding_value_pcm2 = matching_rows['Text_ortho'].values[0]
                        else:
                            corresponding_value_pcm2 = ''  # Placeholder if arg2 not found in treebank
                #check with arg2 first
                else:

                
                    # Step 2: Search for arg1 in treebank['Text_en']
                    matching_rows = treebank.loc[treebank['Text_en'].str.contains(a2_raw, case=False, na=False, regex=False)]
                    if not matching_rows.empty:
                        corresponding_value_pcm1 = matching_rows['Text_ortho'].values[0]

                        # Step 3: Search for conn1 + arg1 in treebank['Text_en']
                        conn_arg2_text = c_raw + a1_raw
                        matching_rows = treebank.loc[treebank['Text_en'].str.contains(conn_arg2_text, case=False, na=False, regex=False)]
                        if not matching_rows.empty:
                            corresponding_value_pcm2 = matching_rows['Text_ortho'].values[0]
                        else:
                            # Step 4: Search for arg2 in treebank['Text_en']
                            matching_rows = treebank.loc[treebank['Text_en'].str.contains(a1_raw, case=False, na=False, regex=False)]
                            if not matching_rows.empty:
                                corresponding_value_pcm2 = matching_rows['Text_ortho'].values[0]
                            else:
                                corresponding_value_pcm2 = ''  # Placeholder if arg2 not found in treebank

            
                # Combine pcm1 and pcm2 and append to pcm_full_text
                pcmText = corresponding_value_pcm1 + " " + corresponding_value_pcm2
                pcm_full_text.append(pcmText)

    return pcm_full_text


treebank = pd.read_csv("TreeBankAnnotated/csv/conllu/allDataFramesConllu.csv")
df = pd.read_csv("TreeBankAnnotated/csv/processed/mergedAnnotationFullText.csv")

fulltext = df['full_text']
arg1 = df['arg1raw']
arg2 = df['arg2raw']
conn_raw = df['conn_raw']

result = search_treebank(treebank, fulltext, arg1, arg2, conn_raw)
print(len(result))
print(len(fulltext))

df['PCM_FULL_TEXT'] = result
df.to_csv("TreeBankAnnotated/csv/processed/mergedTextWithPCMFullText.csv")
dfnng = df[df['PCM_FULL_TEXT'] != "neglect for now"]
dfnng.to_csv("TreeBankAnnotated/csv/processed/mergedTextWithPCMFullTextNFN.csv")


df = pd.read_csv("TreeBankAnnotated/csv/processed/mergedTextWithPCMFullText.csv")





pcmFullText = df['PCM_FULL_TEXT'].to_list()

englishToPred = ["translate pcm to english: "+ line for line in pcmFullText]

t5Preds = model.predict(englishToPred)

df['EnglishTranslationPCM'] = t5Preds
