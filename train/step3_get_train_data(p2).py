import json
import os
import ast
import openpyxl

import pandas as pd

from sklearn.utils import shuffle


def read_wiki_data():
    # 定義檔案資料夾路徑和檔案名稱的格式
    data_folder = r"wiki\\Pre_Data"
    wiki_name = "pre-wiki-0{}.jsonl"

    # 讀取所有 JSONL 檔案，並轉換為字典物件
    wiki = []
    for i in range(24):
        if i == 0:
            continue
        if i < 10:
            filename = os.path.join(
                data_folder, wiki_name.format("0" + str(i)))
        else:
            filename = os.path.join(data_folder, wiki_name.format(i))
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                wiki.append(line)

    return wiki


def search_wiki(id):
    if result := list(filter(lambda x: id == x['id'], wiki)):
        return result[0]['evidence_list']
    else:
        return None


def read_excel(filename):
    wb = openpyxl.load_workbook(filename)
    sheet = wb['Sheet1']
    rows = sheet.rows
    header = [cell.value for cell in next(rows)]
    data = []
    for row in rows:
        values = [cell.value for cell in row]
        data.append(dict(zip(header, values)))
    return data


def shuffle_data(data):
    return shuffle(data)


public_train = read_excel('data/public_train.xlsx')
step2 = read_excel('data/step2(public_train).xlsx')

wiki = read_wiki_data()

index = 0
output = []

for data in step2:
    index += 1

    print(f'{index}/{len(step2)}')

    id = data['id']
    claim = data['claim']
    label = data['label']

    evidence_list = ast.literal_eval(data['evidence_list'])
    evidence_string_list = []

    if label == 0:
        for evidence in evidence_list:
            # evidence[0] : id
            # evidence[1] : index
            if search_result := search_wiki(evidence[0]):
                if result := list(filter(lambda x: evidence[1] == x['index'], search_result)):
                    result = result[0]
                    if result['evidence'] in evidence_string_list:
                        continue

                    if len(result['evidence']) <= 3:
                        continue
                    evidence_string_list.append(result['evidence'])
    else:
        evidence_temp = []

        for evidence in evidence_list:
            # evidence[0] : id
            # evidence[1] : index
            evidence_temp.append({
                'id': evidence[0],
                'index': evidence[1]
            })

        check = False

        if public_train_data := list(filter(lambda x: id == x['id'], public_train)):
            public_train_data = public_train_data[0]
            public_train_data['evidence'] = ast.literal_eval(
                public_train_data['evidence'])
            for check_evidence in public_train_data['evidence']:
                if list(filter(lambda x: check_evidence['id'] == x['id'] and check_evidence['index'] == x['index'], evidence_temp)):
                    check = True
                    break

        if check:
            pass
        else:
            continue

        for evidence in evidence_list:
            # evidence[0] : id
            # evidence[1] : index
            if search_result := search_wiki(evidence[0]):
                if result := list(filter(lambda x: evidence[1] == x['index'], search_result)):
                    result = result[0]
                    if result['evidence'] in evidence_string_list:
                        continue

                    if len(result['evidence']) <= 3:
                        continue
                    evidence_string_list.append(result['evidence'])

    if len(evidence_string_list) == 0:
        continue

    if list(filter(lambda x: id == x['id'], output)):
        continue

    evidence_1 = 0
    evidence_2 = 0
    evidence_3 = 0
    evidence_4 = 0
    evidence_5 = 0

    if len(evidence_string_list) >= 1:
        evidence_1 = evidence_string_list[0]
    if len(evidence_string_list) >= 2:
        evidence_2 = evidence_string_list[1]
    if len(evidence_string_list) >= 3:
        evidence_3 = evidence_string_list[2]
    if len(evidence_string_list) >= 4:
        evidence_4 = evidence_string_list[3]
    if len(evidence_string_list) >= 5:
        evidence_5 = evidence_string_list[4]

    output.append({
        'id': id,
        'label': label,
        'claim': claim,
        'evidence_1': evidence_1,
        'evidence_2': evidence_2,
        'evidence_3': evidence_3,
        'evidence_4': evidence_4,
        'evidence_5': evidence_5
    })

# 資料同類別過於集中，使用shuffle_data重新排序
output = shuffle_data(output)
pd.DataFrame(output).to_excel(
    'data/step3_train_data_p2.xlsx', index=False)
