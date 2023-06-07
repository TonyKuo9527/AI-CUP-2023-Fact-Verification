import json
import wikipedia
import opencc
import openpyxl
import requests
import ast
import os
import pandas as pd

wikipedia.set_lang("zh")

CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")


api_key = ''  # openai api key


def get_keywords_openai_prompt1_ex(claim):
    messages = []

    Prompt1 = '執行以下步驟:\n'
    Prompt1 += '1.閱讀陳述句\n'
    Prompt1 += '2.分析陳述句內容\n'
    Prompt1 += '3.擷取句子中完整關鍵字\n'

    Prompt1 += '==='

    Prompt1 += '必須使用list格式輸出，以較重要的三個關鍵字組作為值'

    Prompt1 += '==='

    Prompt1 += '陳述句:\n'
    Prompt1 += claim + '。'

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


def get_keywords_openai_prompt2_ex(claim):
    messages = []

    Prompt2 = '執行以下步驟:\n'
    Prompt2 += '1.閱讀陳述句\n'
    Prompt2 += '2.分析陳述句內容\n'
    Prompt2 += '3.擷取句子中完整關鍵詞組\n'

    Prompt2 += '==='

    Prompt2 += '使用list格式輸出，以較重要的三個關鍵詞組作為值'

    Prompt2 += '==='

    Prompt2 += '陳述句:\n'
    Prompt2 += claim + '。'

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


def get_keywords_openai_prompt_final(claim, P1, P2):
    messages = []

    Prompt = '執行以下步驟:\n'
    Prompt += '1.閱讀陳述句\n'
    Prompt += '2.分析陳述句內容\n'
    Prompt += '3.擷取句子中完整關鍵字\n'

    Prompt += '==='

    Prompt += '必須使用list格式輸出，以較重要的三個關鍵字組作為值'

    Prompt += '==='

    Prompt += '陳述句:\n'
    Prompt += claim + '。\n'

    Prompt += '==='

    Prompt += '已驗證無效關鍵字:\n'

    predictions = []

    for evidence in P1:
        if evidence in predictions:
            pass
        else:
            predictions.append(evidence)

    for evidence in P2:
        if evidence in predictions:
            pass
        else:
            predictions.append(evidence)

    for evidence in predictions:
        Prompt += evidence + '\n'

    messages.append({'role': 'system', 'content': Prompt})

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
            print(response['choices'][0]['message']['content'])
            return response['choices'][0]['message']['content']
    except:
        return None


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


def do_st_corrections(text):
    simplified = CONVERTER_T2S.convert(text)

    return CONVERTER_S2T.convert(simplified)


def get_wiki_title(text):
    try:
        return do_st_corrections(wikipedia.search(text)[0])
    except:
        return None


def search_wiki(id):
    if result := list(filter(lambda x: id == x['id'], wiki)):
        return result[0]['id']
    else:
        return None


def search_wiki_sp(id):

    if result := list(filter(lambda x:
                             (id == x['id']) or
                             (id == x['id'].replace('·', '')) or
                             (id == x['id'].replace('-', '')) or
                             (id == x['id'].replace(' ', '')) or
                             (id == x['id'].replace('_', ' ')) or
                             (id == x['id'].split('_(')[0]), wiki)):
        return result
    else:
        return None


def processing(P1, P2):
    keywords = []
    predictions = []
    verify = []

    for data in P1:
        if data not in keywords:
            keywords.append(data)

    for data in P2:
        if data not in keywords:
            keywords.append(data)

    temp = []

    for keyword in keywords:
        if result := get_wiki_title(keyword):
            temp.append(result)

    for data in temp:
        if data not in keywords:
            keywords.append(data)

    for keyword in keywords:
        if results := search_wiki_sp(keyword):
            for result in results:
                if result['id'] not in predictions:
                    predictions.append(result['id'])

    for prediction in predictions:
        if result := search_wiki(prediction):
            if result not in verify:
                verify.append(result)

    return verify


wiki = read_wiki_data()

DataSet = read_excel('data/participle(public_train).xlsx')

index = 0
output = []

for data in DataSet:
    index += 1

    print(f'{index}/{len(DataSet)}')

    id = data['id']
    label = data['label']
    claim = data['claim']
    evidence_id = data['evidence_id']
    evidence = data['evidence']

    if data['P1']:
        try:
            P1 = ast.literal_eval(data['P1'])
        except:
            P1 = []
    else:
        P1 = []

    if data['P2']:
        try:
            P2 = ast.literal_eval(data['P2'])
        except:
            P2 = []
    else:
        P2 = []

    if P1 == []:
        if P1 := get_keywords_openai_prompt1_ex(claim):
            try:
                P1 = ast.literal_eval(P1)
            except:
                P1 = []
        else:
            P1 = []

    if P2 == []:
        if P2 := get_keywords_openai_prompt2_ex(claim):
            try:
                P2 = ast.literal_eval(P2)
            except:
                P2 = []
        else:
            P2 = []

    predictions = processing(P1, P2)

    if predictions == []:
        if final := get_keywords_openai_prompt_final(claim, P1, P2):
            try:
                final = ast.literal_eval(final)
            except:
                final = []
        else:
            final = []

        predictions = processing(final, final)

    output.append({
        'id': id,
        'label': label,
        'claim': claim,
        'evidence_id': evidence_id,
        'evidence': evidence,
        'predictions': predictions
    })

    if (index % 100) == 0:  # 避免程式中斷時，可以從中斷的地方繼續執行
        df = pd.DataFrame(output)
        df.to_excel(
            'data/evidence_id_predictions(public_train).xlsx', index=False)

df = pd.DataFrame(output)
df.to_excel(
    'data/evidence_id_predictions(public_train).xlsx', index=False)
