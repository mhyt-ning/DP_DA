# Required imports
import torch, gc
from Teacher import Teacher
from data import load_teacher_data, NoisyDataset
from util import accuracy, split
from Student import Student
# import syft as sy
# import syft.frameworks.torch.dp.pate as pate
from my_framework import pate
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from autodp.mechanism_zoo import GaussianMechanism
from autodp.mechanism_zoo import LaplaceMechanism
from autodp.calibrator_zoo import eps_delta_calibrator
import numpy as np
import pdb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from data import TextDataset

target_epsilon = '1'
print(f'target_epsilon={target_epsilon}')


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
        self.sigma = 21.1
        self.save_path = './model/model_' + str(target_epsilon)


args = Arguments()
delta = 1e-6
partition1 = 0.23

class PATE(GaussianMechanism):
    def __init__(self, sigma, m, Binary, name='PATE'):
        # sigma is the std of the Gaussian noise added to the voting scores
        if Binary:
            # This is a binary classification task
            sensitivity = 1
        else:  # for the multiclass case, the L2 sensitivity is sqrt(2)
            sensitivity = np.sqrt(2)
        GaussianMechanism.__init__(self, sigma=sigma / sensitivity / np.sqrt(m), name=name)

        self.params = {'sigma': sigma}


teacher_n=Teacher(args,n_teachers=args.n_teachers,is_init=0,sigma=args.sigma)
t_train_loader,t_test_loader, gen_loader= load_teacher_data(n_teachers=args.n_teachers, batch_size=args.batchsize,target_epsilon=target_epsilon)



#____________________student________________________
print("\n")
print("\n")
print("Training Student")


student_path = 'train_test/eps'+str(target_epsilon)+'/student_data.csv'
student_data = pd.read_csv(student_path).sample(frac=1, random_state=None)
student_texts = student_data['input'].tolist()
student_labels = student_data['output'].tolist()

for i in range(len(student_texts)):
    student_texts[i] = str(student_texts[i]).lower()
    if str(student_labels[i]) == 'True':
        student_labels[i] = 1
    else:
        student_labels[i] = 0

student_dataset = TextDataset(student_texts, student_labels)
student_loader = DataLoader(student_dataset, args.batchsize, shuffle=True)


s_train_loader, s_test_loader = split(student_loader, split=partition1)



num=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i, batch in enumerate(s_train_loader):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label'].to(device)

    for k in range(len(labels)):
        print(labels[k])
        if labels[k]==1:
            num+=1



m=num
print(f'len(s_train_loader): {len(s_train_loader) * args.batchsize}; len(s_test_loader):{len(s_test_loader) * args.batchsize}')

student = Student(args)
N = NoisyDataset(s_train_loader, teacher_n.predict)
student.train(N)

results = []
targets = []

total = 0.0
correct = 0.0

for batch in t_test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    predict_lol = student.predict(input_ids,attention_mask).to(device)
    predict_lol=torch.max(predict_lol, 1)[1]
    correct += float((predict_lol == (labels)).sum().item())
    total += float(labels.size(0))

print("for teacher_test_data, Private Baseline: ", (correct / total) * 100)

total = 0.0
correct = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for batch in s_test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    predict_lol = student.predict(input_ids, attention_mask).to(device)
    predict_lol = torch.max(predict_lol, 1)[1]
    correct += float((predict_lol == (labels)).sum().item())
    total += float(labels.size(0))

    results.extend(predict_lol.cpu().numpy())
    targets.extend(labels.cpu().numpy())

print("Private Baseline: ", (correct / total) * 100)
s_accuracy = accuracy_score(targets, results)
s_report = classification_report(targets,results)

print("for Student_test:")
print(f'Student, Accuracy: {s_accuracy:.4f}')
print(s_report)


pate_mech = PATE(sigma=args.sigma, m=m, Binary=True, name='PATE')
eps = pate_mech.get_approxDP(delta)

print(f'teacher num:{args.n_teachers},sigma={args.sigma}')
print(f'access num:{m}')
print('autodp: ')
print(f'eps:{eps},delta:{delta}')
print(f'save: {args.save_path}')