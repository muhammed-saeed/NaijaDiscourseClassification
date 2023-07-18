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
model_path = "/local/musaeed/CLaTCheckpointsWithoutDev/checkpoint-65268-epoch-9"
model = T5Model("t5", model_path, args=model_args)

df = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/processed/mergedTextWithPCMFullTextAndTranslated.csv")
pcmFullText = df['PCM_FULL_TEXT'].to_list()

englishToPredNoDEVTEst= ["translate pcm to english: "+ line for line in pcmFullText]

t5Preds = model.predict(englishToPredNoDEVTEst)

df['EnglishTranslationPCMWithoutDEVTest'] = t5Preds

df.to_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/processed/mergedTreeBankAnnotationWithoutDevTest.csv")
