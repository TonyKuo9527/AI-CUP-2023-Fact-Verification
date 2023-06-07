import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 設定隨機種子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_input(claim, evidence_id, evidence, tokenizer, max_length):
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
        'attention_mask': inputs['attention_mask'][0]
    }

    return input_dict


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        claim = self.data[index]['claim']
        evidence_id = self.data[index]['id']
        evidence = self.data[index]['evidence']
        label = self.data[index]['label']

        inputs = preprocess_input(
            claim, evidence_id, evidence, self.tokenizer, self.max_length)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.int64)
        }


def load_data(filename, tokenizer, max_length, test_size=0.15, val_size=0.15, batch_size=16):
    df = pd.read_excel(filename)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=val_size/(1-test_size), random_state=42)

    train_dataset = MyDataset(train_df.to_dict(
        'records'), tokenizer, max_length)
    val_dataset = MyDataset(val_df.to_dict('records'), tokenizer, max_length)
    test_dataset = MyDataset(test_df.to_dict('records'), tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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


# 設定模型超參數
max_length = 256
hidden_dim = 256
output_dim = 2
num_epochs = 5
batch_size = 90
lr = 7e-5

model_name = 'bert-base-chinese'

file_name = 'data/step2_train_data.xlsx'


tokenizer = BertTokenizer.from_pretrained(model_name)
print('model_name: {}'.format(model_name))
print('batch_size: {}, learning_rate: {}'.format(
    batch_size, lr))
# 加載數據集
train_loader, val_loader, test_loader = load_data(
    file_name, tokenizer, max_length, test_size=0.1, val_size=0.1, batch_size=batch_size)

# 定義模型
bert = BertModel.from_pretrained(model_name)
model = BertClassifier(
    bert, bert.config.hidden_size, hidden_dim, output_dim)
model.to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# ======================== #


patience = 2  # 決定要等待多少個epoch，如果驗證損失沒有改善就提前停止訓練
early_stop = 0  # 用來追蹤驗證損失多少個epoch沒有改善
early_epoch = 0  # 用來保存最早獲得最佳驗證損失的epoch
best_val_f1 = 0

# 增加一行程式碼來保存在驗證集上表現最好的模型權重
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    print('Epoch: {}/{}'.format(epoch + 1, num_epochs))

    running_loss = 0.0
    running_f1 = 0.0
    model.train()
    for batch in train_loader:
        train_batch_size = len(batch['input_ids'])

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * train_batch_size
        running_f1 += f1_score(labels.cpu().numpy(), logits.argmax(
            dim=1).cpu().numpy(), average='macro') * train_batch_size

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_f1 = running_f1 / len(train_loader.dataset)

    print('Train Loss: {:.4f} | F1: {:.4f}'.format(epoch_loss, epoch_f1))

    # 驗證模型
    val_running_loss = 0.0
    val_running_f1 = 0.0
    model.eval()
    for val_batch in val_loader:
        val_batch_size = len(val_batch['input_ids'])

        input_ids = val_batch['input_ids'].to(device)
        attention_mask = val_batch['attention_mask'].to(device)
        labels = val_batch['labels'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_running_loss += loss.item() * val_batch_size
            val_running_f1 += f1_score(labels.cpu().numpy(),
                                       logits.argmax(dim=1).cpu().numpy(), average='macro') * val_batch_size

    val_loss = val_running_loss / len(val_loader.dataset)
    val_f1 = val_running_f1 / len(val_loader.dataset)
    print('Val Loss: {:.4f} | F1: {:.4f}'.format(val_loss, val_f1))

    # 檢查驗證損失是否有改善
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_epoch = epoch
        early_stop = 0

        # 如果驗證損失有改善，則保存當前的模型權重
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        early_stop += 1
        if early_stop >= patience:
            print('Early stopping.')
            break  # 如果驗證損失沒有改善，則提前停止訓練

# 在訓練結束時，將模型的權重設定為在驗證集上表現最好的權重
model.load_state_dict(best_model_wts)

# 計算測試集的F1分數
model.eval()
y_true = []
y_pred = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy())

test_f1 = f1_score(y_true, y_pred, average='macro')
print('Test F1: {:.4f}'.format(test_f1))

output_model_name = 'model/step2({}_{}_{}).pt'.format(
    max_length, batch_size, lr)

torch.save(model.state_dict(), output_model_name)
