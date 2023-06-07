import torch
import openpyxl
import torch.nn as nn
import torch.nn.functional as F
import json
import pandas as pd
import os
import ast

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


def preprocess_input(claim, evidence_id, evidence, tokenizer, max_length):
    text = f'{claim}[SEP]{evidence_id}[SEP]{evidence}'

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )

    input_dict = {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'token_type_ids': inputs['token_type_ids'][0]
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

model_path = 'model/step2(256_90_7e-05).pt'
bert = BertModel.from_pretrained(model_name)
model = BertClassifier(bert, bert.config.hidden_size, 256, 2)
model.load_state_dict(torch.load(model_path))

max_length = 256
model.to(device)


def predict(data):
    model.eval()
    data = preprocess_input(
        data['claim'], data['evidence_id'], data['evidence'], tokenizer, max_length)
    with torch.no_grad():
        input_ids = data['input_ids'].unsqueeze(0).to(device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(device)
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()

        return probabilities[0][1]


wiki = read_wiki_data()
DataSet = read_excel('data/evidence_id_predictions(public_test).xlsx')

index = 0
output = []

for data in DataSet:
    index += 1

    print(f'{index} / {len(DataSet)}')

    id = data['id']
    claim = data['claim']
    predictions = ast.literal_eval(data['predictions'])
    predict_list = []

    for prediction in predictions:
        if wiki_evidence_list := search_wiki(prediction):
            for wiki_evidence in wiki_evidence_list:
                if probability := predict({'claim': claim, 'evidence_id': prediction, 'evidence': wiki_evidence['evidence']}):
                    predict_list.append({
                        'id': prediction,
                        'index': wiki_evidence['index'],
                        'probability': probability
                    })

    predict_list = sorted(
        predict_list, key=lambda x: x['probability'], reverse=True)

    top_five = predict_list[:5]

    evidence = []

    for top in top_five:
        evidence.append([top['id'], top['index'], top['probability']])

    output.append({
        'id': id,
        'claim': claim,
        'evidence': evidence,
    })

df = pd.DataFrame(output)
df.to_excel('data/step2(public_test).xlsx', index=False)
