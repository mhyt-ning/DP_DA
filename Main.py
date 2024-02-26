# Required imports
import torch,gc
from Teacher import Teacher
from data import load_data, NoisyDataset
from util import accuracy, split
from Student import Student
import syft as sy
# import syft.frameworks.torch.dp.pate as pate
from my_framework import pate

from autodp.mechanism_zoo import GaussianMechanism 
from autodp.mechanism_zoo import LaplaceMechanism
from autodp.calibrator_zoo import eps_delta_calibrator
import numpy as np


target_epsilon=4
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
        self.n_teachers = 100
        self.save_model = False
        self.sigma=6
        self.save_path='./model/model_test_'+str(target_epsilon)
        # self.save_path='./model/model_laplace_4'

args = Arguments()
delta = 1e-6
partition=0.05

class PATE(GaussianMechanism):
    def __init__(self,sigma, m, Binary,name='PATE'):
        # sigma is the std of the Gaussian noise added to the voting scores
        if Binary:
            # This is a binary classification task
            sensitivity = 1
        else: # for the multiclass case, the L2 sensitivity is sqrt(2)
            sensitivity = np.sqrt(2)
        GaussianMechanism.__init__(self, sigma=sigma/sensitivity/np.sqrt(m),name=name)
        
        self.params = {'sigma':sigma}

class PATE2(LaplaceMechanism):
    def __init__(self,b, m, Binary,name='Laplace'):
        # sigma is the std of the Gaussian noise added to the voting scores
        if Binary:
            # This is a binary classification task
            sensitivity = 1
        else: # for the multiclass case, the L2 sensitivity is sqrt(2)
            sensitivity = np.sqrt(2)
        LaplaceMechanism.__init__(self, b=b/sensitivity/np.sqrt(m),name=name)
        
        self.params = {'b':b}



# calibrate = eps_delta_calibrator()
# calibrate2 = eps_delta_calibrator()


# train_loaders,test_loaders,test_loader,test_texts, test_labels = load_data(n_teachers=args.n_teachers,batch_size=args.batchsize)

train_loaders,test_loaders,test_loader = load_data(n_teachers=args.n_teachers,batch_size=args.batchsize)
test_train, test_val = split(test_loader,split=partition)
teacher_n=Teacher(args,n_teachers=args.n_teachers,is_init=0,sigma=args.sigma)

outputs=[]
outputs2=[]
# Declare and train teachers on MNIST training data
for i in range(args.n_teachers):
    teacher = Teacher(args,n_teachers=1,sigma=args.sigma)
    output,output2=teacher.train(i,train_loaders,test_loaders,test_train,test_val)
    outputs.append(output)
    outputs2.append(output2)

# gc.collect()
# torch.cuda.empty_cache()

# # Evaluate Teacher accuracy
teacher_targets = []
predict = []
predict_nonoise=[]
counts = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i,batch in enumerate(test_train):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label'].to(device)
    teacher_targets.append(labels)
    
    mid_output=[]
    for j in range(args.n_teachers):
        mid_output.append(outputs[j][i])

    noisy_output = teacher_n.predict(input_ids,attention_mask,mid_output)
    predict.append(noisy_output["predictions"])
    predict_nonoise.append(noisy_output["predictions_nonoise"])
    counts.append(noisy_output["model_counts"])

print("对于test_train")
print("教师模型聚合未加噪时Accuracy: ", accuracy(predict_nonoise, teacher_targets))
print("教师模型聚合加噪后Accuracy: ", accuracy(predict, teacher_targets))


# # Evaluate Teacher accuracy
teacher_targets2 = []
predict2 = []
predict_nonoise2=[]
counts2 = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i,batch in enumerate(test_val):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label'].to(device)
    teacher_targets2.append(labels)
    
    mid_output=[]
    for j in range(args.n_teachers):
        mid_output.append(outputs2[j][i])

    noisy_output2 = teacher_n.predict(input_ids,attention_mask,mid_output)
    predict2.append(noisy_output2["predictions"])
    predict_nonoise2.append(noisy_output2["predictions_nonoise"])
    counts2.append(noisy_output2["model_counts"])

print("对于test_val")
print("教师模型聚合未加噪时Accuracy: ", accuracy(predict_nonoise2, teacher_targets2))
print("教师模型聚合加噪后Accuracy: ", accuracy(predict2, teacher_targets2))


print("\n")
print("\n")

print("Training Student")

# Split the test data further into training and validation data for student
print(f'学生训练集大小：{len(test_train)*args.batchsize}；学生测试集大小：{len(test_val)*args.batchsize}')

student = Student(args)
N = NoisyDataset(test_train, teacher_n.predict,predict)
student.train(N)

results = []
targets = []

total = 0.0
correct = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for batch in test_val:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    predict_lol = student.predict(input_ids,attention_mask).to(device)
    predict_lol=torch.max(predict_lol, 1)[1]
    correct += float((predict_lol == (labels)).sum().item())
    total += float(labels.size(0))

print("Private Baseline: ", (correct / total) * 100)


# counts_lol = torch.stack(counts).contiguous().view(args.n_teachers, -1)
# predict_lol = torch.tensor(predict).view(len(counts_lol[1]))
# print("Laplace噪声:")
# # data_dep_eps, data_ind_eps = teacher.analyze(counts_lol, predict_lol, moments=5)
# print("Epsilon: ", teacher.analyze(counts_lol, predict_lol,moments=8))

m = len(test_train)*args.batchsize


pate_mech = PATE(sigma=args.sigma,m=m,Binary=True, name='PATE')
eps = pate_mech.get_approxDP(delta)


# pate_mech2 = PATE2(b=1.23,m=m,Binary=True, name='Laplace')
# eps2 = pate_mech2.get_approxDP(delta)

# print(f'教师数量:{args.n_teachers},b=0.5')
print(f'教师数量:{args.n_teachers},sigma={args.sigma}')
print(f'学生访问教师次数:{m}')
print('autodp: ')
print(f'eps:{eps},delta:{delta}')
# print(f'eps:{eps2},delta:{delta}')
