import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM,RobertaForSequenceClassification, AdamW, AutoModelForSequenceClassification,AutoModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import random
import re
from tqdm import tqdm
import string
import numpy as np

# 假设的数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = tokenizer(text, max_length=512, truncation=True, padding='max_length')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
import pandas as pd
import json
# 打开并读取JSON文件

# 指定子集的总大小
total_subset_size = 3750

#选择baseline
baseline='myway_0.1'
print(baseline)
source_dataset_path = 'result/baselines/'+baseline+'.csv'  # 源数据集的路径
save_path='result/baselines/'+baseline+'_'+str(total_subset_size)+'.csv'
data = pd.read_csv(source_dataset_path).sample(frac=1, random_state=42).reset_index(drop=True)

true_probabilities = [prob for prob in data["probabilities"]]
for i in range(len(true_probabilities)):
    a=true_probabilities[i].replace(']', ' ')
    a=" ".join(a.split())
    a=a.split(' ')[1]
    true_probabilities[i]=float(a)

data["true_probabilities"]=true_probabilities

with open('count_addnoise_0.4.json', 'r') as file:
    count_addnoise = json.load(file)
# 计算总频率
total_freq=0
for key in count_addnoise:
    total_freq+=count_addnoise[key]
# 将频率转换为概率
target_distribution = {key: freq / total_freq for key, freq in count_addnoise.items()}

# 根据目标分布计算每个类别应选取的数量
# 注意，这里我们先计算一个初步的数量，后续可能需要调整以确保总大小符合要求
preliminary_counts = {k: int(v * total_subset_size) for k, v in target_distribution.items()}

# 确保总数加起来等于total_subset_size，可能需要调整某些类别的数量
adjustment = total_subset_size - sum(preliminary_counts.values())
for k in sorted(preliminary_counts, key=lambda x: target_distribution[x], reverse=True):
    if adjustment == 0:
        break
    preliminary_counts[k] += 1
    adjustment -= 1

# 选择每个类别中True概率最高的前n_i个样本
selected_samples = pd.DataFrame()
for feature_value, count in preliminary_counts.items():
    # 筛选特定类别的样本，并按照True概率降序排序
    sorted_samples = data[data['medical_specialty'] == feature_value].sort_values(by='true_probabilities', ascending=False)
    
    # 选择True概率最高的前n_i个样本
    top_samples = sorted_samples.head(count)
    
    # 将这些样本加入到最终选择的样本集中
    selected_samples = pd.concat([selected_samples, top_samples])

selected_samples=selected_samples.sample(frac=1, random_state=42).reset_index(drop=True)

selected_samples.to_csv(save_path, index=False)

train_path=save_path

test_path='task_data/test_data.csv'

train_data = pd.read_csv(train_path).sample(frac=1, random_state=42)

train_texts = train_data['input'].tolist()
train_labels = train_data['medical_specialty'].tolist()

test_data = pd.read_csv(test_path).sample(frac=1, random_state=42)
test_texts = test_data['transcription'].tolist()
test_labels = test_data['medical_specialty'].tolist()


batchsize=64
num1 =len(train_data)
num2= len(test_data)
#80 50 50
num_epochs = 80
num_iterate=3


label_mapping = {'Chiropractic': 0,
 'Urology': 1,
 'Psychiatry(Psychology)': 2,
 'Office Notes': 3,
 'Autopsy': 4,
 'Radiology': 5,
 'IME-QME-Work Comp etc.': 6,
 'Emergency Room Reports': 7,
 'Pediatrics - Neonatal': 8,
 'Discharge Summary': 9,
 'Dermatology': 10,
 'Neurology': 11,
 'Diets and Nutritions': 12,
 'Speech - Language': 13,
 'Rheumatology': 14,
 'Orthopedic': 15,
 'Podiatry': 16,
 'Hospice - Palliative Care': 17,
 'Allergy(Immunology)': 18,
 'Bariatrics': 19,
 'Pain Management': 20,
 'Cosmetic(Plastic Surgery)': 21,
 'Gastroenterology': 22,
 'Hematology - Oncology': 23,
 'Surgery': 24,
 'Consult - History and Phy.': 25,
 'Letters': 26,
 'Nephrology': 27,
 'Cardiovascular(Pulmonary)': 28,
 'Dentistry': 29,
 'SOAP(Chart(Progress Notes)': 30,
 'Physical Medicine - Rehab': 31,
 'Ophthalmology': 32,
 'Neurosurgery': 33,
 'Lab Medicine - Pathology': 34,
 'Sleep Medicine': 35,
 'General Medicine': 36,
 'ENT - Otolaryngology': 37,
 'Endocrinology': 38,
 'Obstetrics(Gynecology)': 39}


for i in range(len(train_texts)):
    train_texts[i] = str(train_texts[i]).lower()
    train_labels[i] = label_mapping[str(train_labels[i])]


for i in range(len(test_texts)):
    test_texts[i] = str(test_texts[i]).lower()
    test_labels[i] = label_mapping[str(test_labels[i])]
    

accuracies=[]
for i in range(num_iterate):
    train_dataset = TextDataset(train_texts, train_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    tokenizer = AutoTokenizer.from_pretrained('../private-transformers/examples/huggingface/medical/clinicalbert')
    model = AutoModelForSequenceClassification.from_pretrained("../private-transformers/examples/huggingface/medical/clinicalbert",
                                                            num_labels=40)


    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)


    # 模型训练与评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-6)

    max_accuracy=-1
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)

                predictions.extend(predicted_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        max_accuracy=max(accuracy,max_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}')
        print(report)
    
    accuracies.append(max_accuracy)

average_accuracy = np.mean(accuracies)
print(accuracies)
print(f"平均准确度: {average_accuracy}")
print(f'epochs数:{num_epochs}')
print(f'数据集：{train_path},测试集：{test_path}')
print(f'训练集数量：{num1}, 测试集数量：{num2}')