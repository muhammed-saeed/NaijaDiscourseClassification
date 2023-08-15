import json
import csv
import os

folder_path = '/local/musaeed/discopy/realEnglish'  # replace with the path to your folder
output_csv = '/local/musaeed/jsonSample/RealEnglishDiscopyPrased.csv'

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
                        'Connective_RawText': 'nan',
                        'Connective_CharacterSpanList': 'nan',
                        'Sense': 'nan',
                        'Type': 'nan'
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
