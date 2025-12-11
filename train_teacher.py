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


#____________________teacher________________________
t_train_loader,t_test_loader, gen_loader= load_teacher_data(n_teachers=args.n_teachers, batch_size=args.batchsize,target_epsilon=target_epsilon)



print("\n")
print("\n")
print("Training Teachers")

outputs = []
outputs2 = []
# Declare and train teachers on MNIST training data
for i in range(args.n_teachers):
    teacher = Teacher(args, n_teachers=1, sigma=args.sigma)
    output, output2 = teacher.train(i, t_train_loader, t_test_loader, gen_loader)
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
    # print("Labels in batch", i, ":", labels)
    
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
        # print(same_count/len(labels))
        # print(result_array[j])
    # print("teacher result for batch", i)
    # print(total_right/(len(labels)*args.n_teachers))

    noisy_output = teacher_n.predict(input_ids,attention_mask,mid_output)
    noisy_outputs2.append(noisy_output)
    predict2.append(noisy_output["predictions"])
    predict_nonoise2.append(noisy_output["predictions_nonoise"])
    counts2.append(noisy_output["model_counts"])

print("for teacher-test")
print("no_noisy_Accuracy: ", accuracy(predict_nonoise2, teacher_targets2))
print("after_noisy_Accuracy: ", accuracy(predict2, teacher_targets2))


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
        text = tokenizer.decode(input_ids[k], skip_special_tokens=True)
        need_texts.append(text)
        probability_list.append(probability_distribution[i][k])

texts_probability = pd.DataFrame({
    'text': need_texts,
    'probability': probability_list
})
texts_probability.to_csv('train_test/eps'+str(target_epsilon)+'/texts_probability.csv',index=False)
print("texts_probability save")

texts_probability=pd.read_csv('train_test/eps'+str(target_epsilon)+'/texts_probability.csv').sample(frac=1, random_state=42).reset_index(drop=True)

true_probabilities = [prob for prob in texts_probability["probability"]]
for i in range(len(true_probabilities)):
    a=true_probabilities[i].replace(']', ' ')
    a=" ".join(a.split())
    a=a.split(' ')[1]
    true_probabilities[i]=float(a)

texts_probability["true_probability"]=true_probabilities


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

print("finish")

