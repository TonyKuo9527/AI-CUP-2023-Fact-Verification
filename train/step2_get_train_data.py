import openpyxl
import ast
import os
import json
import numpy as np
import pandas as pd


def read_wiki_data():
    # 定義檔案資料夾路徑和檔案名稱的格式
    data_folder = r"wiki\\Pre_Data"
    wiki_name = "pre-wiki-0{}.jsonl"

    # 讀取所有 JSONL 檔案，並轉換為字典物件
    wiki = []
    for i in range(25):
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


def get_evidence(id, index):
    if wiki_data := search_wiki(id):
        if result := list(filter(lambda x: index == x['index'], wiki_data)):
            return result[0]['evidence']
        else:
            return None
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


def get_evidence_index(claim, evidence_id):
    if result := list(filter(lambda x: claim == x['claim'] and (evidence_id in x['evidence']), public_train)):
        return result[0]['evidence']
    else:
        return None


def random_evidence(data):
    if len(data) == 0:
        return None
    else:
        evidence_string = ''
        while len(evidence_string) < 5:
            evidence_string = data[np.random.randint(0, len(data))]['evidence']

        return evidence_string


wiki = read_wiki_data()
public_train = read_excel('data\public_train.xlsx')
evidence_id_predictions = read_excel(
    'data\evidence_id_predictions(public_train).xlsx')
index = 0
output = []

for data in evidence_id_predictions:
    index += 1
    print(f'{index} / {len(evidence_id_predictions)}')

    id = data['id']
    claim = data['claim']
    label = data['label']

    try:
        predictions = ast.literal_eval(data['predictions'])
    except:
        predictions = []

    if predictions == []:
        continue

    if label == 0:
        continue

    count = 0

    public_train_data = list(filter(lambda x: id == x['id'], public_train))[0]
    evidence_id_list = ast.literal_eval(public_train_data['evidence_id'])
    evidence_index_list = ast.literal_eval(public_train_data['evidence'])

    for evidence in evidence_index_list:
        if evidence_string := get_evidence(evidence['id'], evidence['index']):
            count += 1
            output.append({
                'id': id,
                'label': 1,
                'claim': claim,
                'evidence_id': evidence['id'],
                'evidence': evidence_string,
            })

    for evidence_id in predictions:
        if count == 0:
            break

        if evidence_id in evidence_id_list:
            continue

        if results := search_wiki(evidence_id):
            result = random_evidence(results)

            if result is None:
                continue

            count -= 1
            output.append({
                'id': id,
                'label': 0,
                'claim': claim,
                'evidence_id': evidence_id,
                'evidence': result,
            })

    if index % 100 == 0:
        df = pd.DataFrame(output)
        df.to_excel('data/step2_train_data.xlsx', index=False)

df = pd.DataFrame(output)
df.to_excel('data/step2_train_data.xlsx', index=False)
