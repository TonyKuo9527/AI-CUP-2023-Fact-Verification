import torch
import openpyxl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
import os
import json

from transformers import BertTokenizer, BertModel


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


def preprocess_input_p1(claim, evidence_id, evidence, tokenizer, max_length):
    text = f'{claim}[SEP]{evidence_id}[SEP]{evidence}'

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_dict = {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
    }

    return input_dict


def preprocess_input_p2(claim, evidence_list, tokenizer, max_length):
    text = f'{claim}'

    for evidence in evidence_list:
        text += f'[SEP]{evidence["evidence"]}'

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_dict = {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
    }

    return input_dict


class FCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BertClassifier(nn.Module):
    def __init__(self, bert, input_dim, hidden_dim, output_dim):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.fc = FCClassifier(input_dim, hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = bert_outputs.last_hidden_state[:, 0, :]
        logits = self.fc(last_hidden_state_cls)
        return logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)

bert_p1 = BertModel.from_pretrained(model_name)
p1 = 'model/step3_p1(256_90_7e-05).pt'
p1_model = BertClassifier(bert_p1, bert_p1.config.hidden_size, 256, 2)
p1_model.load_state_dict(torch.load(p1))
p1_model.to(device)

bert_p2 = BertModel.from_pretrained(model_name)
p2 = 'model/step3_p2(512_16_2e-05).pt'
p2_model = BertClassifier(bert_p2, bert_p2.config.hidden_size, 256, 3)
p2_model.load_state_dict(torch.load(p2))
p2_model.to(device)


def predict_p1(data):
    p1_model.eval()
    data = preprocess_input_p1(
        data['claim'], data['id'], data['evidence'], tokenizer, 256)
    with torch.no_grad():
        input_ids = data['input_ids'].unsqueeze(0).to(device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(device)
        logits = p1_model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        probabilities = probabilities[0].tolist()

        return torch.argmax(logits, dim=1).cpu().numpy()[0], probabilities


def predict_p2(data):
    p2_model.eval()
    data = preprocess_input_p2(
        data['claim'], data['evidence_list'], tokenizer, 512)
    with torch.no_grad():
        input_ids = data['input_ids'].unsqueeze(0).to(device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(device)
        logits = p2_model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        probabilities = probabilities[0].tolist()

        return torch.argmax(logits, dim=1).cpu().numpy()[0], probabilities


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


def insert_data(data):
    with open('result/private_test.jsonl', 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


wiki = read_wiki_data()
DataSet = read_excel('data/step2(private_test).xlsx')

index = 0
output = []

for data in DataSet:
    index += 1
    print(f'{index}/{len(DataSet)}')

    claim = data['claim']
    id = data['id']
    evidence_list = ast.literal_eval(data['evidence'])

    predicted_evidence = []
    evidence_string_list = []

    label = 0

    for evidence in evidence_list:
        if evidence_string := get_evidence(evidence[0], evidence[1]):
            if evidence[2] < 0.3:
                continue

            predict_label, probabilities = predict_p1({
                'claim': claim,
                'id': evidence[0],
                'evidence': evidence_string
            })

            predicted_evidence.append([evidence[0], evidence[1]])

            evidence_string_list.append({
                'evidence': evidence_string
            })

            if predict_label == 1:
                label = 1

    if label == 0:
        predicted_label = 'NOT ENOUGH INFO'
        predicted_evidence = None
    else:
        predict_label, probabilities = predict_p2({
            'claim': claim,
            'evidence_list': evidence_string_list
        })

        if predict_label == 0:
            predicted_label = 'NOT ENOUGH INFO'
            predicted_evidence = None
        elif predict_label == 1:
            predicted_label = 'SUPPORTS'
        elif predict_label == 2:
            predicted_label = 'REFUTES'

    insert_data({
        'id': id,
        'predicted_label': predicted_label,
        'predicted_evidence': predicted_evidence,
    })
