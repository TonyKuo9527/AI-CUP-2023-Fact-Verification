import json
import openpyxl
import requests
import pandas as pd

api_key = ''  # openai api key


def get_keywords_openai_prompt1(claim):
    messages = []

    Prompt1 = '輸入一段句子，分析句子內容擷取完整關鍵字。\n'
    Prompt1 += '輸出格式為陣列，以較為重要的三筆作為值。\n'

    messages.append({'role': 'system', 'content': Prompt1})

    messages.append({'role': 'user', 'content': claim})

    # 設置請求內容
    # 使用requests而非openai，因為openai若高頻率使用，中間容易發生錯誤
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'model': 'gpt-3.5-turbo',
        'messages': messages,
        'temperature': 0.2,
        'max_tokens': 200,
        'top_p': 0.5,
    }

    json_data = json.dumps(data)
    try:
        response = requests.post(url, headers=headers, data=json_data)
        if (response.status_code == 200):
            response = response.json()
            return response['choices'][0]['message']['content']
    except:
        return None


def get_keywords_openai_prompt2(claim):
    messages = []

    Prompt2 = '輸入一段句子，分析句子內容擷取完整關鍵詞組。\n'
    Prompt2 += '輸出格式為陣列，以較為重要的三筆作為值。\n'

    messages.append({'role': 'system', 'content': Prompt2})

    messages.append({'role': 'user', 'content': claim})

    # 設置請求內容
    # 使用requests而非openai，因為openai若高頻率使用，中間容易發生錯誤
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        'model': 'gpt-3.5-turbo',
        'messages': messages,
        'temperature': 0.2,
        'max_tokens': 200,
        'top_p': 0.5,
    }

    json_data = json.dumps(data)
    try:
        response = requests.post(url, headers=headers, data=json_data)
        if (response.status_code == 200):
            response = response.json()
            return response['choices'][0]['message']['content']
    except:
        return None


def extract_list_from_string(string):
    start = string.find('[')
    end = string.rfind(']')
    if start != -1 and end != -1:
        lst_str = string[start+1:end]
        lst = [item.strip().strip('"') for item in lst_str.split(',')]
        return lst
    else:
        return []


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


DataSet = read_excel('data/public_train.xlsx')

index = 0

output = []

for data in DataSet:
    index += 1

    print(f'{index} / {len(DataSet)}')

    id = data['id']
    label = data['label']
    claim = data['claim']
    evidence_id = data['evidence_id']
    evidence = data['evidence']

    if openai_output_p1 := get_keywords_openai_prompt1(claim):
        openai_output_p1 = extract_list_from_string(openai_output_p1)

    if openai_output_p2 := get_keywords_openai_prompt2(claim):
        openai_output_p2 = extract_list_from_string(openai_output_p2)

    output.append({
        'id': id,
        'label': label,
        'claim': claim,
        'evidence_id': evidence_id,
        'evidence': evidence,
        'P1': openai_output_p1,
        'P2': openai_output_p2,
    })

    if index % 20 == 0:
        df = pd.DataFrame(output)
        df.to_excel('data/participle(public_train).xlsx', index=False)

df = pd.DataFrame(output)
df.to_excel('data/participle(public_train).xlsx', index=False)
