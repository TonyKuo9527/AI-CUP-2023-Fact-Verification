import json
import pandas as pd


public_train_data = []
public_train_0316 = 'wiki\\public_train_0316.jsonl'
with open(public_train_0316, 'r', encoding='utf-8') as f:
    for data in f:
        data = json.loads(data)

        evidence_id = []
        evidence_list = []

        if data['label'] == 'NOT ENOUGH INFO':
            label = 0
            data['evidence'] = []
        elif data['label'] == 'supports':
            label = 1
            data['evidence'] = data['evidence'][0]
        elif data['label'] == 'refutes':
            label = 2
            data['evidence'] = data['evidence'][0]

        for evidence in data['evidence']:
            evidence_list.append({
                'id': evidence[2],
                'index': evidence[3]
            })
            if evidence[2] in evidence_id:
                pass
            else:
                evidence_id.append(evidence[2])

        public_train_data.append({
            'id': data['id'],
            'label': label,
            'claim': data['claim'],
            'evidence_id': evidence_id,
            'evidence': evidence_list
        })


public_train_0522 = 'wiki\\public_train_0522.jsonl'
with open(public_train_0522, 'r', encoding='utf-8') as f:
    for data in f:
        data = json.loads(data)

        evidence_id = []
        evidence_list = []

        if data['label'] == 'NOT ENOUGH INFO':
            label = 0
            data['evidence'] = []
        elif data['label'] == 'supports':
            label = 1
            data['evidence'] = data['evidence'][0]
        elif data['label'] == 'refutes':
            label = 2
            data['evidence'] = data['evidence'][0]

        for evidence in data['evidence']:
            evidence_list.append({
                'id': evidence[2],
                'index': evidence[3]
            })
            if evidence[2] in evidence_id:
                pass
            else:
                evidence_id.append(evidence[2])

        public_train_data.append({
            'id': data['id'],
            'label': label,
            'claim': data['claim'],
            'evidence_id': evidence_id,
            'evidence': evidence_list
        })

pd.DataFrame(public_train_data).to_excel(
    'data/public_train.xlsx', index=False)
