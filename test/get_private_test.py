import json
import pandas as pd


private_test_data = []
private_test_data_path = 'wiki\\private_test_data.jsonl'
with open(private_test_data_path, 'r', encoding='utf-8') as f:
    for data in f:
        data = json.loads(data)
        private_test_data.append({
            'id': data['id'],
            'claim': data['claim']
        })

pd.DataFrame(private_test_data).to_excel(
    'data/private_test.xlsx', index=False)
