import pandas as pd

df1 = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/conllu/pcm_nsc_ud_dev_conllu.csv")
# df2 = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/conllu/pcm_nsc_ud_train_conllu.csv")
df3 = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/conllu/pcm_nsc_ud_test_conllu.csv")

total = pd.concat([df1, df3])
total.to_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/conllu/DevTestDataFramesConllu.csv")