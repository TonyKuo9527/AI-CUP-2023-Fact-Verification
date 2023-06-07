import json
import pandas as pd


private_test_data = []
private_test_data_path = 'wiki\\public_test.jsonl'
with open(private_test_data_path, 'r', encoding='utf-8') as f:
    for data in f:
        data = json.loads(data)
        private_test_data.append({
            'id': data['id'],
            'claim': data['claim']
        })

pd.DataFrame(private_test_data).to_excel(
    'data/public_test.xlsx', index=False)
