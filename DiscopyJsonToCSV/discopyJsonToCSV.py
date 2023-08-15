import json
import csv
import os
import argparse

def main(args):
    folder_path = args.folder_path
    output_csv = args.output_csv

    # This is the header for our CSV file
    headers = ['file_index', 'text', 'Arg1_RawText', 'Arg1_CharacterSpanList', 'Arg2_RawText', 'Arg2_CharacterSpanList', 'Connective_RawText', 'Connective_CharacterSpanList', 'Sense', 'Type']

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
        csv_writer.writeheader()

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith('.json'):
                file_index = int(file_name.split('.')[0])  # Extracting the file index from the filename
                
                with open(os.path.join(folder_path, file_name), 'r') as json_file:
                    data = json.load(json_file)

                    text = data['text']
                    relations = data['relations']

                    # If there are no relations, write NaNs
                    if not relations:
                        csv_writer.writerow({
                            'file_index': file_index,
                            'text': text,
                            'Arg1_RawText': 'nan',
                            'Arg1_CharacterSpanList': 'nan',
                            'Arg2_RawText': 'nan',
                            'Arg2_CharacterSpanList': 'nan',
                            'Connective_RawText': 'Missing',
                            'Connective_CharacterSpanList': 'nan',
                            'Sense': 'Missing',
                            'Type': 'Missing'
                        })

                    # If there are relations, write each relation to a new row in the CSV
                    for relation in relations:
                        csv_writer.writerow({
                            'file_index': file_index,
                            'text': text,
                            'Arg1_RawText': relation['Arg1']['RawText'],
                            'Arg1_CharacterSpanList': relation['Arg1']['CharacterSpanList'],
                            'Arg2_RawText': relation['Arg2']['RawText'],
                            'Arg2_CharacterSpanList': relation['Arg2']['CharacterSpanList'],
                            'Connective_RawText': relation['Connective']['RawText'],
                            'Connective_CharacterSpanList': relation['Connective']['CharacterSpanList'],
                            'Sense': ','.join(relation['Sense']),
                            'Type': relation['Type']
                        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and create a CSV file.")
    parser.add_argument("--folder_path", type=str, default='/local/musaeed/discopy/fake', help="Path to the folder containing JSON files")
    parser.add_argument("--output_csv", type=str, default='/local/musaeed/jsonSample/translatedEnglishDiscopyPrased.csv', help="Path to the output CSV file")

    args = parser.parse_args()
    main(args)