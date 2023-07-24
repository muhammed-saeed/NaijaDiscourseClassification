import argparse
import pandas as pd

def extract_lines_with_prefix(file_path, prefix):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    prefix_lines = []
    for line in lines:
        if line.startswith(prefix):
            prefix_lines.append(line.strip())

    return prefix_lines

def extract_value(line):
    return line.split(' = ')[-1]

def main(file_path, output_csv):
    # Extract lines starting with #sound_url
    sound_url_lines = extract_lines_with_prefix(file_path, '# sound_url')
    sound_url_values = [extract_value(line) for line in sound_url_lines]

    # Extract lines starting with #text
    text_lines = extract_lines_with_prefix(file_path, '# text')
    text_values = [extract_value(line) for line in text_lines]

    # Extract lines starting with #text_en
    text_en_lines = extract_lines_with_prefix(file_path, '# text_en')
    text_en_values = [extract_value(line) for line in text_en_lines]

    # Extract lines starting with #text_ortho
    text_ortho_lines = extract_lines_with_prefix(file_path, '# text_ortho')
    text_ortho_values = [extract_value(line) for line in text_ortho_lines]

    # Extract lines starting with #speaker_id
    speaker_id_lines = extract_lines_with_prefix(file_path, '# speaker_id')
    speaker_id_values = [extract_value(line) for line in speaker_id_lines]

    # Extract lines starting with #sent_id
    sent_id_lines = extract_lines_with_prefix(file_path, '# sent_id')
    sent_id_values = [extract_value(line) for line in sent_id_lines]

    # Print the extracted values
    print('Speaker IDs:')
    print(len(speaker_id_values))
    print()

    print('Texts:')
    print(len(text_values))
    print()

    print('Text_en values:')
    print(len(text_en_values))
    print()

    print('Text_ortho values:')
    print(len(text_ortho_values))
    print()

    print('Sent IDs:')
    print(len(sent_id_values))


    print("Sound url")
    print(len(sound_url_values))


    # Create a dictionary with the extracted lists
    data = {
        'Speaker ID': speaker_id_values,
        'Text_en': text_en_values,
        'Text_ortho': text_ortho_values,
        'Sent ID': sent_id_values,
        'Sound Url': sound_url_values
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print('CSV file saved successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert conllu dataset to CSV.')
    parser.add_argument('--file_path', type=str, default="TreeBankAnnotated/conllu/pcm_nsc-ud-dev.conllu", help='Path to the input conllu file')
    parser.add_argument('--output_csv', type=str, default="TreeBankAnnotated/csv/conllu/pcm_nsc_ud_dev_conllu.csv", help='Path to the output CSV file')
    args = parser.parse_args()

    main(args.file_path, args.output_csv)
