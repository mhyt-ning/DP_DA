# Required imports
import torch, gc
from Teacher import Teacher
from data import load_data, NoisyDataset
from util import accuracy, split
from Student import Student
import syft as sy
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

# import argparse

# parser = argparse.ArgumentParser(description="使用指定参数运行训练。")
# parser.add_argument('--target_epsilon', type=str, default='4', help='训练数据路径')
# parser.add_argument('--n_teachers', type=int, default=15)

# args = parser.parse_args()

target_epsilon = '4'
print(f'target_epsilon={target_epsilon}')

n_teachers=15

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
        self.n_teachers = n_teachers
        self.save_model = False
        self.sigma = 6
        self.save_path = './model/model_' + str(target_epsilon)
        # self.save_path='./model/model_laplace_4'


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


class PATE2(LaplaceMechanism):
    def __init__(self, b, m, Binary, name='Laplace'):
        # sigma is the std of the Gaussian noise added to the voting scores
        if Binary:
            # This is a binary classification task
            sensitivity = 1
        else:  # for the multiclass case, the L2 sensitivity is sqrt(2)
            sensitivity = np.sqrt(2)
        LaplaceMechanism.__init__(self, b=b / sensitivity / np.sqrt(m), name=name)

        self.params = {'b': b}


teacher_n=Teacher(args,n_teachers=args.n_teachers,is_init=0,sigma=args.sigma)


#____________________教师阶段________________________
t_train_loader,t_test_loader, gen_loader, student_loader= load_data(n_teachers=args.n_teachers, batch_size=args.batchsize)
s_train_loader, s_test_loader = split(student_loader, split=partition1)



print("\n")
print("\n")
print("Training Teachers")

outputs = []
outputs2 = []
# Declare and train teachers on MNIST training data
for i in range(args.n_teachers):
    teacher = Teacher(args, n_teachers=1, sigma=args.sigma)
    output, output2 = teacher.train(i, t_train_loader, t_test_loader, gen_loader,s_test_loader)
    outputs.append(output)
    outputs2.append(output2)


tokenizer = AutoTokenizer.from_pretrained('../model/roberta-large')

teacher_targets2 = []
predict2 = []
predict_nonoise2=[]
counts2 = []
noisy_outputs2=[]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i,batch in enumerate(t_test_loader):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label'].to(device)
    teacher_targets2.append(labels)
    # print("outputs in batch", i, ":", outputs)
    
    mid_output=[]
    for j in range(args.n_teachers):
        mid_output.append(outputs2[j][i])
    # print("Mid outputs:", mid_output)
    total_right=0
    result_array = [[0] * len(labels) for _ in range(args.n_teachers)]
    for j in range(len(mid_output)):
        same_count=0
        for k in range(len(mid_output[j])):
            prob1 = mid_output[j][k][0]
            prob2 = mid_output[j][k][1]
            if prob2 > prob1:
                result_array[j][k]=1
            else:
                result_array[j][k]=0
            if result_array[j][k] == labels[k]:
                same_count += 1 
        total_right=total_right+same_count
        # print(same_count/len(outputs))
        # print(result_array[j])
    # print("teacher result for batch", i)
    # print(total_right/(len(outputs)*args.n_teachers))

    noisy_output = teacher_n.predict(input_ids,attention_mask,mid_output)
    noisy_outputs2.append(noisy_output)
    predict2.append(noisy_output["predictions"])
    predict_nonoise2.append(noisy_output["predictions_nonoise"])
    counts2.append(noisy_output["model_counts"])
    # print(f'teacher_targets2:{teacher_targets2[i]}')
    # print(f'predict2:{predict2[i]}')
    # print(f'predict_nonoise2:{predict_nonoise2[i]}')

print("对于teacher-test")
print("教师模型聚合未加噪时Accuracy: ", accuracy(predict_nonoise2, teacher_targets2))
print("教师模型聚合加噪后Accuracy: ", accuracy(predict2, teacher_targets2))

# breakpoint()  # 设置断点

# # Evaluate Teacher accuracy
need_texts=[]
probability_list = []

teacher_targets = []
predict = []
predict_nonoise = []
counts = []
probability_distribution=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i, batch in enumerate(gen_loader):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label'].to(device)
    teacher_targets.append(labels)


    mid_output = []
    for j in range(args.n_teachers):
        mid_output.append(outputs[j][i])

    noisy_output = teacher_n.predict(input_ids, attention_mask, mid_output)
    predict.append(noisy_output["predictions"])
    predict_nonoise.append(noisy_output["predictions_nonoise"])
    counts.append(noisy_output["model_counts"])
    probability_distribution.append(noisy_output["probability_distribution"])


    for k in range(len(input_ids)):
        # 如果需要将input_ids转换回文本，可以调用tokenizer.decode
        text = tokenizer.decode(input_ids[k], skip_special_tokens=True)
        need_texts.append(text)
        probability_list.append(probability_distribution[i][k])

texts_probability = pd.DataFrame({
    'text': need_texts,
    'probability': probability_list
})
texts_probability.to_csv('train_test/eps'+str(target_epsilon)+'/texts_probability.csv',index=False)
print("texts_probability中的内容保存成功")
# breakpoint()  # 设置断点

texts_probability=pd.read_csv('train_test/eps'+str(target_epsilon)+'/texts_probability.csv').sample(frac=1, random_state=42).reset_index(drop=True)

true_probabilities = [prob for prob in texts_probability["probability"]]
for i in range(len(true_probabilities)):
    a=true_probabilities[i].replace(']', ' ')
    a=" ".join(a.split())
    a=a.split(' ')[1]
    true_probabilities[i]=float(a)

texts_probability["true_probability"]=true_probabilities

# 筛选特定类别的样本，并按照True概率降序排序
k=100
sorted_samples = texts_probability.sort_values(by='true_probability', ascending=False)
top_samples = sorted_samples.head(k)
top_samples=top_samples.reset_index(drop=True)
print(top_samples)

label_a = ["True" for _ in range(len(top_samples))]
data1=pd.DataFrame()
data1['input'] = top_samples['text']
data1['output']=label_a
print(data1)

student_path='train_test/student_false_data.csv'
student_data=pd.read_csv(student_path).sample(frac=1, random_state=42)
student_data=student_data[:100]
data2=pd.DataFrame()
data2['input'] =student_data['input']
data2['output'] =student_data['output']
df=pd.concat([data1,data2], ignore_index=True).sample(frac=1, random_state=42)
for i in range(len(df)):
    df.loc[i,'input']=str(df.loc[i,'input']).lower()

df.to_csv('train_test/eps'+str(target_epsilon)+'/student_data.csv',index=False)

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

#____________________学生阶段________________________
print("\n")
print("\n")
print("Training Student")

num=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i, batch in enumerate(s_train_loader):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label'].to(device)

    for k in range(len(labels)):
        print(labels[k])
        # 如果需要将input_ids转换回文本，可以调用tokenizer.decode
        if labels[k]==1:
            num+=1



m=num
print(f'学生训练集大小：{len(s_train_loader) * args.batchsize}；学生测试集大小：{len(s_test_loader) * args.batchsize}')

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

print("学生对于teacher_test, Private Baseline: ", (correct / total) * 100)

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

print("对于Student_test:")
print(f'Student, Accuracy: {s_accuracy:.4f}')
print(s_report)

# counts_lol = torch.stack(counts).contiguous().view(args.n_teachers, -1)
# predict_lol = torch.tensor(predict).view(len(counts_lol[1]))
# print("Laplace噪声:")
# # data_dep_eps, data_ind_eps = teacher.analyze(counts_lol, predict_lol, moments=5)
# print("Epsilon: ", teacher.analyze(counts_lol, predict_lol,moments=8))

# m = len(s_train_loader) * args.batchsize

pate_mech = PATE(sigma=args.sigma, m=m, Binary=True, name='PATE')
eps = pate_mech.get_approxDP(delta)

# pate_mech2 = PATE2(b=1.23,m=m,Binary=True, name='Laplace')
# eps2 = pate_mech2.get_approxDP(delta)

# print(f'教师数量:{args.n_teachers},b=0.5')
print(f'教师数量:{args.n_teachers},sigma={args.sigma}')
print(f'学生访问教师次数:{m}')
print('autodp: ')
print(f'eps:{eps},delta:{delta}')
print(f'学生模型存储路径：{args.save_path}')
# print(f'eps:{eps2},delta:{delta}')
