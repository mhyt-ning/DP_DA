import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import random
import re
from tqdm import tqdm
import string
import numpy as np
from torch.nn.functional import softmax
import ast

# Required imports
import torch,gc
from Teacher import Teacher
from data import load_data, NoisyDataset
from util import accuracy, split
from Student import Student
# import syft as sy
# import syft.frameworks.torch.dp.pate as pate
from my_framework import pate
import random
from data import TextDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset


c_path='task_data/task_data_3.5.csv'
target_epsilon=0.1
print(f'target_epsilon:{target_epsilon}')
save_path='../classification/result/baselines/myway_'+str(target_epsilon)+'.csv'


choose_data= pd.read_csv(c_path)

# threshold = 0.5
accuracies = []  # 存储每次迭代的准确度


class Arguments:

    # Class used to set hyperparameters for the whole PATE implementation
    def __init__(self):
        self.batchsize = 8
        self.test_batchsize = 10
        self.epochs = 10
        self.student_epochs = 20
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.n_teachers = 15
        self.save_model = False
        # self.sigma=0.77
        self.save_path='./model/model_'+str(target_epsilon)

args = Arguments()

student = Student(args)
student.load_models()
model=student

# 应用模型到完整的测试集
full_test_texts = [str(text) for text in choose_data["input"].tolist()]
full_test_dataset = TextDataset(full_test_texts, [0] * len(full_test_texts))  # 用0填充标签
full_test_loader = DataLoader(full_test_dataset, batch_size=32, shuffle=False)


test_probabilities = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with torch.no_grad():
    for batch in full_test_loader:     
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model.predict(input_ids, mask=attention_mask)
        probabilities = softmax(logits, dim=1).cpu().numpy()
        test_probabilities.extend(probabilities)

# 确保概率列表的长度与 DataFrame 行数相同
if len(test_probabilities) == len(choose_data):
    choose_data['probabilities'] = test_probabilities
else:
    print("概率列表长度与 DataFrame 行数不匹配。")


# true_probabilities = [prob[1] for prob in choose_data["probabilities"]]
# filtered_data = choose_data[[prob > threshold for prob in true_probabilities]]

# print(f"设定的筛选阈值为:{threshold}")
print(f"预测以下数据集的分类概率：{c_path}")
print('数据集大小：',len(choose_data))
# print("筛选出的样本数：", len(filtered_data))
print(f"result存储路径：{save_path}")
# 将筛选出的样本保存到CSV文件
choose_data.to_csv(save_path, index=False)
