import json
import openpyxl
import requests
import pandas as pd

api_key = ''  # openai api key


def get_keywords_openai_prompt1(claim):
    messages = []

    Prompt1 = '執行以下步驟:\n'
    Prompt1 += '1.閱讀陳述句\n'
    Prompt1 += '2.分析陳述句內容\n'
    Prompt1 += '3.擷取句子中完整關鍵字\n'

    Prompt1 += '==='

    Prompt1 += '使用list格式輸出，以較重要的三個關鍵字組作為值'

    Prompt1 += '==='

    Prompt1 += '陳述句:\n'
    Prompt1 += claim + '\n'

    messages.append({'role': 'system', 'content': Prompt1})

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

    Prompt2 = '執行以下步驟:\n'
    Prompt2 += '1.閱讀陳述句\n'
    Prompt2 += '2.分析陳述句內容\n'
    Prompt2 += '3.擷取句子中完整關鍵詞組\n'

    Prompt2 += '==='

    Prompt2 += '使用list格式輸出，以較重要的三個關鍵詞組作為值'

    Prompt2 += '==='

    Prompt2 += '陳述句:\n'
    Prompt2 += claim + '\n'

    messages.append({'role': 'system', 'content': Prompt2})

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


DataSet = read_excel('data/public_test.xlsx')

index = 0

output = []

for data in DataSet:
    index += 1

    print(f'{index} / {len(DataSet)}')

    id = data['id']
    claim = data['claim']

    openai_output_p1 = None
    openai_output_p2 = None

    count = 0

    while count < 3:
        openai_output_p1 = get_keywords_openai_prompt1(claim)
        if openai_output_p1 is not None:
            count = 0
            break
        else:
            count += 1

    while count < 3:
        openai_output_p2 = get_keywords_openai_prompt2(claim)
        if openai_output_p2 is not None:
            count = 0
            break
        else:
            count += 1

    if openai_output_p1 is None:
        openai_output_p1 = []

    if openai_output_p2 is None:
        openai_output_p2 = []

    output.append({
        'id': id,
        'claim': claim,
        'P1': openai_output_p1,
        'P2': openai_output_p2,
    })

    if index % 20 == 0:
        df = pd.DataFrame(output)
        df.to_excel('data/participle(public_test).xlsx', index=False)

df = pd.DataFrame(output)
df.to_excel('data/participle(public_test).xlsx', index=False)
