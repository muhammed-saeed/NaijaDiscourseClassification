import pandas as pd 
real_df = pd.read_csv("/local/musaeed/jsonSample/RealEnglishDiscopyPrased.csv")
translated_df = pd.read_csv("/local/musaeed/jsonSample/translatedEnglishDiscopyPrased.csv")
real_df = real_df.sort_values(by='file_index')
translated_df = translated_df.sort_values(by='file_index')
real_df.to_csv("/local/musaeed/jsonSample/SortedRealEnglishDiscopyPrased.csv", index=False)
translated_df.to_csv("/local/musaeed/jsonSample/SortedTranslatedEnglishDiscopyPrased.csv",  index=False)