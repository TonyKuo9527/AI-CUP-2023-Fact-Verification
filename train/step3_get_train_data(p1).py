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

for data in public_train:
    index += 1

    print(f'{index}/{len(public_train)}')

    if index == 500:
        break

    id = data['id']
    claim = data['claim']
    label = data['label']

    evidence_list = ast.literal_eval(data['evidence'])

    for evidence in evidence_list:
        if search_result := search_wiki(evidence['id']):
            if result := list(filter(lambda x: evidence['index'] == x['index'], search_result)):
                result = result[0]
                if list(filter(lambda x: id == x['id'] and result['evidence'] == x['evidence'], output)):
                    continue
                output.append({
                    'id': id,
                    'label': 1,
                    'claim': claim,
                    'evidence_id': evidence['id'],
                    'evidence': result['evidence']
                })

    count = 0

    if label == 0:
        if step2_evidence_list := list(
                filter(lambda x: id == x['id'], step2)):
            step2_evidence_list = step2_evidence_list[0]
            for step2_evidence in ast.literal_eval(step2_evidence_list['evidence_list']):
                if count == 5:
                    break

                if search_result := search_wiki(step2_evidence[0]):
                    if result := list(filter(lambda x: step2_evidence[1] == x['index'], search_result)):
                        result = result[0]

                        if len(result['evidence']) < 3:
                            continue

                        count += 1

                        output.append({
                            'id': id,
                            'label': 0,
                            'claim': claim,
                            'evidence_id': step2_evidence[0],
                            'evidence': result['evidence']
                        })

# 資料同類別過於集中，使用shuffle_data重新排序
output = shuffle_data(output)
pd.DataFrame(output).to_excel(
    'data/step3_train_data_p1.xlsx', index=False)
