import os
import json


def insert_data(name, data):
    with open(name, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


data_folder = r'wiki\\Row_Data'
pre_data_folder = r'wiki\\Pre_Data'
wiki_name = 'wiki-0{}.jsonl'
pre_wiki_name = 'pre-wiki-0{}.jsonl'


for i in range(24):  # 處理24份wiki row data
    if i + 1 == 0:
        continue
    if i + 1 < 10:
        file_name = os.path.join(
            data_folder, wiki_name.format('0' + str(i + 1)))
        pre_file_name = os.path.join(
            pre_data_folder, pre_wiki_name.format('0' + str(i + 1)))
    else:
        file_name = os.path.join(data_folder, wiki_name.format(str(i + 1)))
        pre_file_name = os.path.join(
            pre_data_folder, pre_wiki_name.format(str(i + 1)))

    print(f'Processing {file_name}...')

    with open(file_name, 'r', encoding='utf-8') as f:
        index = 0
        for data in f:
            index += 1
            data = json.loads(data)
            data_lines = data['lines'].strip().split('\n')
            evidence_list = []
            for line in data_lines:
                line = line.split('\t')

                if len(line) == 1:
                    continue

                try:
                    line[0] = int(line[0])
                except:
                    line[0] = None
                    print(f'Error: {file_name} {index}')

                if line[1]:
                    evidence_list.append({
                        'index': line[0],
                        'evidence': line[1]
                    })

            output = {
                'index': int(index),
                'id': data['id'],
                'evidence_list': evidence_list
            }

            insert_data(pre_file_name, output)
