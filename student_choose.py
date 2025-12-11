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
from data import NoisyDataset
from util import accuracy, split
from Student import Student
# import syft as sy
# import syft.frameworks.torch.dp.pate as pate
from my_framework import pate
import random
from data import TextDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset


#model_15_
c_path='task_data/gen_data_student.csv'

target_epsilon='1'
print(f'target_epsilon:{target_epsilon}')
# print("laplace")
save_path='../roberta_classification_medical/result/baselines/myway_'+str(target_epsilon)+'.csv'


batchsize=64

# save_path='../roberta_classification/result/baselines/myway_laplace.csv'


choose_data= pd.read_csv(c_path)

# threshold = 0.5
accuracies = []  



class Arguments:

    # Class used to set hyperparameters for the whole PATE implementation
    def __init__(self):
        self.batchsize = 16
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


full_test_texts = [str(text) for text in choose_data["input"].tolist()]
full_test_dataset = TextDataset(full_test_texts, [0] * len(full_test_texts))  
full_test_loader = DataLoader(full_test_dataset, batch_size=batchsize, shuffle=False)


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


if len(test_probabilities) == len(choose_data):
    choose_data['probabilities'] = test_probabilities
else:
    print("no match")



print(f'save_path:{args.save_path}')
print(f"c_path: {c_path}")
print('data size:',len(choose_data))
print(f"result save: {save_path}")
choose_data.to_csv(save_path, index=False)
