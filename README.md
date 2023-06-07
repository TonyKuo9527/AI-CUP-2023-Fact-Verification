# AI CUP 2023 真相只有一個: 事實文字檢索與查核競賽 - Team_3170

## 運行環境
 - Python : 3.11.2
 - Pip : 22.3.1
 - PyTorch : 2.0.0+cu118
 - CUDA : 11.8
 - CUDA Toolkit : 11.8

### 運行裝置規格
 - OS : Windows 10
 - CPU : R9 3900X
 - GPU : RTX3090 Ti 24G
 - Ram : 64G

### 套件版本
 - pandas: 1.5.3
 - openpyxl: 3.1.2
 - requests: 2.28.2
 - wikipedia: 1.4.0
 - opencc: 1.1.1
 - torch: 2.0.0+cu118
 - transformers: 4.27.4
 - sklearn: 0.0.post1 

## Wiki Data
- 下載連結 : [Download](https://drive.google.com/file/d/1hdgMQ2zBiuqGXAKZpdJH4xnANQRlAz-b/view?usp=sharing)
- 由於檔案過大且數量眾多，採用額外下載方式取得，請另行下載解壓縮的方式導入。
- 將下載好的壓縮檔案複製到`train/wiki` & `test/wiki`資料夾內解壓縮，解壓縮後出現的wiki資料夾內的資料為已預處理後的Wiki Data，請務必放置到指定位置，以確保程式能夠正常載入使用。

## 模型權重
|Model|Epoch|Input Max Length|Batch Size|Output|Loss Function|Optimizer|Learning Rate|Direction|URL|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|step2|3|256|90|2|CrossEntropyLoss|AdamW|7e-05|預測陳述句與證據句的相關性|[Download](https://drive.google.com/file/d/1YIC7u4w78QRRRRqhiOATLML4HhjVCQi9/view?usp=sharing)|
|step3_p1|5|256|90|2|CrossEntropyLoss|AdamW|7e-05|預測證據句是否足夠驗證陳述句|[Download](https://drive.google.com/file/d/1rjA2PIyZ-rTZxIXnvjSauOgo6WO7CNai/view?usp=sharing)|
|step3_p2|4|512|16|3|CrossEntropyLoss|AdamW|2e-05|預測陳述句與證據句組合的關係|[Download](https://drive.google.com/file/d/1Lx8ajkk7jHHwz_NbjQ__7FY_lmFvt1Ao/view?usp=sharing)|

- 請將下載好的檔案放置`train/model` & `test/model`資料夾，以確保程式能夠正常載入使用。

## Folder Structure
```
│── test
│   ├── data
│   │   ├── evidence_id_predictions(private_test).xlsx      # private_test資料證據搜尋結果輸出
│   │   ├── evidence_id_predictions(public_test).xlsx       # public_test資料證據搜尋結果輸出
│   │   ├── participle(private_test).xlsx                   # private_test斷詞輸出
│   │   ├── participle(public_test).xlsx                    # public_train斷詞輸出
│   │   ├── private_test.xlsx                               # private_test.jsonl預處理後保存
│   │   ├── public_test.xlsx                                # public_test.jsonl預處理後保存
│   │   ├── step2(private_test).xlsx                        # 輸入evidence_id_predictions(private_test).xlsx，透過step2模型進行陳述句與證據關聯性預測，輸出保存成 step2(private_test).xlsx
│   │   └── step2(public_test).xlsx                         # 輸入evidence_id_predictions(public_test).xlsx，透過step2模型進行陳述句與證據關聯性預測，輸出保存成 step2(public_test).xlsx
│   ├── model
│   │   ├── step2(256_90_7e-05).pt
│   │   ├── step3_p1(256_90_7e-05).pt
│   │   └── step3_p2(512_16_2e-05).pt
│   ├── result
│   │   ├── private_test.jsonl                              # private_test查證結果
│   │   └── public_test.jsonl                               # public_test查證結果
│   ├── wiki                                                # 主辦單位提供之證據資料及預處理後的檔案(由於檔案過大且數量眾多，採用額外下載方式取得)
│   │   ├── Pre_Data                                        # 透過step0_pre_processing_wiki_data.py處理過的資料
│   │   ├── Row_Data                                        # 原始資料
│   │   ├── private_test_data.jsonl
│   │   ├── public_test.jsonl
│   │   ├── public_train_0316.jsonl
│   │   └── public_train_0522.jsonl
│   ├── get_private_test.py                                 # private_test.jsonl預處理後保存
│   ├── get_public_test.py                                  # public_test.jsonl預處理後保存
│   ├── step0_pre_processing_wiki_data.py                   # 處理原始wiki jsonl檔案，輸出對應的Pre_Data便於後續生成、訓練、預測使用
│   ├── step1_participle(private_test).py                   # 透過 Open AI API 以Prompt的方式進行斷詞處理，輸出保存成 participle(private_test).xlsx
│   ├── step1_participle(public_test).py                    # 透過 Open AI API 以Prompt的方式進行斷詞處理，輸出保存成 participle(public_test).xlsx
│   ├── step1_search_evidence_id(private_test).py           # 透過 wiki search & 搜尋wiki_Pre_Data找出相關證據，輸出保存成 evidence_id_predictions(private_test).xlsx
│   ├── step1_search_evidence_id(public_test).py            # 透過 wiki search & 搜尋wiki_Pre_Data找出相關證據，輸出保存成 evidence_id_predictions(public_test).xlsx
│   ├── step2(private_test).py                              # 輸入evidence_id_predictions(private_test).xlsx，透過step2模型進行陳述句與證據關聯性預測，輸出保存成 step2(private_test).xlsx
│   ├── step2(public_test).py                               # 輸入evidence_id_predictions(public_test).xlsx，透過step2模型進行陳述句與證據關聯性預測，輸出保存成 step2(public_test).xlsx
│   ├── step3(private_test).py                              # 輸入step2(private_test).xlsx，透過step3_p1 & step3_p2模型進行陳述句與證據關係預測，輸出保存成 public_test.jsonl
│   └── step3(public_test).py                               # 輸入step2(public_test).xlsx，透過step3_p1 & step3_p2模型進行陳述句與證據關係預測，輸出保存成 public_test.jsonl
│── train
│   ├── data
│   │   ├── evidence_id_predictions(public_train).xlsx      # public_train資料證據搜尋結果輸出
│   │   ├── participle(public_train).xlsx                   # public_train斷詞輸出
│   │   ├── public_train.xlsx                               # public_train_0316.jsonl & public_train_0522.jsonl 預處理後保存
│   │   ├── step2(public_train).xlsx                        # step2預測輸出
│   │   ├── step2_train_data.xlsx                           # 訓練step2模型所使用的資料(由step2_get_train_data.py產出)，為了避免Label 0 資料過多的情況，我們採用隨機採樣證據內的資料而非全部輸出，可能會出現輸出的Label 0資料不同的情況
│   │   ├── step3_train_data_p1.xlsx                        # 訓練step3_p1模型所使用的資料(由step3_get_train_data(p1).py產出)，由於原始資料存在相同類別過度密集分布情況，保存前會進行資料重新排序，因此會出現每次輸出排序有所不同的情況
│   │   └── step3_train_data_p2.xlsx                        # 訓練step3_p2模型所使用的資料(由step3_get_train_data(p2).py產出)，由於原始資料存在相同類別過度密集分布情況，保存前會進行資料重新排序，因此會出現每次輸出排序有所不同的情況
│   ├── model
│   │   └── step2(256_90_7e-05).pt
│   ├── wiki                                                # 主辦單位提供之證據資料及預處理後的檔案(由於檔案過大且數量眾多，採用額外下載方式取得)
│   │   ├── Pre_Data                                        # 透過step0_pre_processing_wiki_data.py處理過的資料
│   │   ├── Row_Data                                        # 原始資料
│   │   ├── private_test_data.jsonl
│   │   ├── public_test.jsonl
│   │   ├── public_train_0316.jsonl
│   │   └── public_train_0522.jsonl
│   ├── step0_get_public_train.py                           # 處理public_train_0316.jsonl & public_train_0522.jsonl，輸出保存成 public_train.xlsx
│   ├── step0_pre_processing_wiki_data.py                   # 處理原始wiki jsonl檔案，輸出對應的Pre_Data便於後續生成、訓練、預測使用
│   ├── step1_participle(public_train).py                   # 透過 Open AI API 以Prompt的方式進行斷詞處理，輸出保存成 participle(public_train).xlsx (該程式的Prompt為初版，考量時間及資源，因此未調整為最終版重新進行斷詞處理)
│   ├── step1_search_evidence_id(public_train).py           # 透過 wiki search & 搜尋wiki_Pre_Data找出相關證據，輸出保存成 evidence_id_predictions(public_train).xlsx 
│   ├── step2(public_train).py                              # 輸入evidence_id_predictions(public_train).xlsx，透過step2模型進行陳述句與證據關聯性預測，輸出保存成 step2(public_train).xlsx
│   ├── step2_get_train_data.py                             # 透過 public_train.xlsx & evidence_id_predictions(public_train).xlsx資料，產出訓練step2模型所需要的訓練資料，輸出保存成 step2_train_data.xlsx ()
│   ├── step2_train_model.py                                # step2模型訓練程式
│   ├── step3_get_train_data(p1).py                         # 透過 public_train.xlsx & step2(train).xlsx資料，產出訓練step3_p1模型所需要的訓練資料
│   ├── step3_get_train_data(p2).py                         # 透過 public_train.xlsx & step2(train).xlsx資料，產出訓練step3_p2模型所需要的訓練資料
│   ├── step3_train_model(p1).py                            # step3_p1模型訓練程式
│   └── step3_train_model(p2).py                            # step3_p2模型訓練程式
│── extra # 補充說明
│   ├── experiment.py                                       # 重現隨機浮動狀況
│   ├── wiki_search.xlsx
└── ...
```
## 執行步驟(Train)
- 下載Wiki Data & 模型權重放置指定資料夾內
- train/step1_participle(public_train).py (Prompt分詞) * 若無Open AI Key建議略過此步驟，直接使用提供的原始檔案
- step1_search_evidence_id(public_train).py (針對Prompt失敗的資料進行額外處理，透過wiki search & 核對wiki Data找出相關證據ID) * 若無Open AI Key建議略過此步驟，直接使用提供的原始檔案
- step2_get_train_data.py (產出訓練step2模型所使用的資料)
- step2_train_model.py (step2模型訓練)
- step2(public_train).py (預測陳述句與證據句的相關性)
- step3_get_train_data(p1).py (產出訓練step3_p1模型所使用的資料)
- step3_get_train_data(p2).py (產出訓練step3_p2模型所使用的資料)
- step3_train_model(p1).py (step3_p1模型訓練)
- step3_train_model(p2).py (step3_p2模型訓練)

## 執行步驟(Test)
- 下載Wiki Data & 模型權重放置指定資料夾內
- test/step1_participle(private_test).py & test/step1_participle(public_test).py (Prompt分詞) * 若無Open AI Key建議略過此步驟，直接使用提供的原始檔案
- step1_search_evidence_id(private_test).py & step1_search_evidence_id(public_test).py (針對Prompt失敗的資料進行額外處理，透過wiki search & 核對wiki Data找出相關證據ID) * 若無Open AI Key建議略過此步驟，直接使用提供的原始檔案
- step2(private_test).py & step2(public_test).py (預測陳述句與證據句的相關性)
- step3(private_test).py & step3(public_test).py (預設陳述句與證據句關係並輸出結果private_test.jsonl & public_test.jsonl)

## 注意事項
- 產出訓練資料部分程式為避免資料類別比例失衡和相同類別資料分布過於密集，程式會隨機指定證據ID資料或是輸出前會進行隨機排序，附上比賽版本所使用的訓練資料供訓練參考。
- 證據查詢使用wikipedia函數庫所提供的"wikipedia.search()"，經過測試後發現，查詢的回傳值存在隨機性(回傳值排序浮動)，可能會導致重新執行證據查詢後的輸出與比賽版本不同，重現查證結果可以直接跳過(Test)的2 & 3步驟。(關於隨機性請參考extra/experiment.py，重現隨機浮動狀況及說明)

## 聯絡信箱
G-Mail : tonykuo9527@gmail.com