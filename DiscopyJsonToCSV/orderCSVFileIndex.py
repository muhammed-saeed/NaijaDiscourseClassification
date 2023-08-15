import pandas as pd
import argparse

def main(args):
    real_df = pd.read_csv(args.real_csv_path)
    translated_df = pd.read_csv(args.translated_csv_path)
    
    real_df = real_df.sort_values(by='file_index')
    translated_df = translated_df.sort_values(by='file_index')
    
    real_df.to_csv(args.sorted_real_csv_path, index=False)
    translated_df.to_csv(args.sorted_translated_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort and save CSV files.")
    parser.add_argument("--real_csv_path", type=str, default="/local/musaeed/jsonSample/RealEnglishDiscopyPrased.csv", help="Path to the original real CSV file")
    parser.add_argument("--translated_csv_path", type=str, default="/local/musaeed/jsonSample/translatedEnglishDiscopyPrased.csv", help="Path to the original translated CSV file")
    parser.add_argument("--sorted_real_csv_path", type=str, default="/local/musaeed/jsonSample/SortedRealEnglishDiscopyPrased.csv", help="Path to save the sorted real CSV file")
    parser.add_argument("--sorted_translated_csv_path", type=str, default="/local/musaeed/jsonSample/SortedTranslatedEnglishDiscopyPrased.csv", help="Path to save the sorted translated CSV file")

    args = parser.parse_args()
    main(args)
