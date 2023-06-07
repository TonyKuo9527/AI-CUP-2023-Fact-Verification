import openpyxl
import wikipedia


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


DataSet = read_excel('wiki_search.xlsx')

index = 0

for data in DataSet:
    index += 1

    print('第' + str(index) + '筆字詞搜尋測試：' + '搜尋字詞:' + data['keyword'])

    check = []  # 檢查是否有浮動隨機回傳結果

    for i in range(1, 15):
        wikipedia.set_lang('zh')
        if i == 1:
            check = wikipedia.search(data['keyword'])
            print(check)
        else:
            result = wikipedia.search(data['keyword'])
            if result == check:
                pass
            else:
                print('出現浮動隨機回傳結果')
                print(result)
                break
