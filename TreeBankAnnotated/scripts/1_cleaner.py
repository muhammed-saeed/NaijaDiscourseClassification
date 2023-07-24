import argparse

def add_missing_text_ortho(file_path, file_path_2):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []
    for i in range(len(lines)):
        line = lines[i].strip()
        updated_lines.append(line)
        if line.startswith('# text_en') and i < len(lines) - 1 and not lines[i + 1].startswith('# text_ortho'):
            updated_lines.append('# text_ortho = XXXXX')

    with open(file_path_2, 'w', encoding='utf-8') as file:
        file.write('\n'.join(updated_lines))

    print('File updated successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find #text_en lines not followed by #text_ortho lines.')
    parser.add_argument('--file_path', type=str, default="TreeBankAnnotated/conllu/pcm_nsc-ud-train.conllu", help='Path to the input conllu file')
    parser.add_argument('--file_path_2', type=str, default="/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/cleanedConllu/pcm_nsc-ud-train_cleaned.conllu", help='Path to the output CSV file')
    args = parser.parse_args()

    add_missing_text_ortho(args.file_path, args.file_path_2)


