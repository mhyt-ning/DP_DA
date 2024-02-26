import torch,gc
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# import syft.frameworks.torch.dp.pate as pate
from my_framework import pate

from transformers import RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
import os

class Teacher:
    """Implementation of teacher models.
       Teacher models are ensemble of models which learns directly disjoint splits of the sensitive data
       The ensemble of teachers are further used to label unlabelled public data on which the student is 
       trained. 
       Args:
           args[Arguments object]: An object of Arguments class with required hyperparameters
           n_teachers[int]: Number of teachers
           epochs[int]: Number of epochs to train each model
    """

    def __init__(self, args, n_teachers=1, is_init=1,sigma=1,epsilon=0.5):

        self.n_teachers = n_teachers
        self.models = {}
        self.args = args
        if is_init==1:
            self.init_models()
        self.epsilon = 100
        self.sigma=sigma

    def init_models(self):
        """Initialize teacher models according to number of required teachers"""

        name = "model_"
        for index in range(0, self.n_teachers):

            model = RobertaForSequenceClassification.from_pretrained('../model/roberta-large',
                                                             num_labels=2)
            self.models[index] = model

    def addnoise(self, x):
        """Adds Laplacian noise to histogram of counts
           Args:
                counts[torch tensor]: Histogram counts
                epsilon[integer]:Amount of Noise
           Returns:
                counts[torch tensor]: Noisy histogram of counts
        """

        # m = Laplace(torch.tensor([0.0]), torch.tensor([self.epsilon]))
        """Adds Gaussian noise to histogram of counts
        """
        m = Normal(torch.tensor([0.0]), torch.tensor([self.sigma]))
        count = x + m.sample()

        return count

    def train(self, train_index,train_loaders,test_loaders,test_train,test_val):
        """Function to train all teacher models.
           Args:
                dataset[torch tensor]: Dataset used to train teachers in format (image,label)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        train_loader=train_loaders[train_index]
        test_loader=test_loaders[train_index]
        
        model=self.models[0]
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        num_epochs = self.args.epochs
        model.train()
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
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
        output=[]
        
        with torch.no_grad():
            for batch in test_train:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                output.append(logits.cpu())

                predicted_labels = torch.argmax(logits, dim=1)
                predictions.extend(predicted_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)

        print("对于test_train:")
        print(f'Teacher {train_index}, Accuracy: {accuracy:.4f}')
        print(report)

        predictions2 = []
        true_labels2 = [] 
        output2=[]
        
        with torch.no_grad():
            for batch in test_val:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                output2.append(logits.cpu())

                predicted_labels = torch.argmax(logits, dim=1)
                predictions2.extend(predicted_labels.cpu().numpy())
                true_labels2.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels2, predictions2)
        report = classification_report(true_labels2, predictions2)

        print("对于test_val:")
        print(f'Teacher {train_index}, Accuracy: {accuracy:.4f}')
        print(report)


        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        return output,output2

    def load_models(self):
    
        path_name = "model_"
    
        for i in range(self.n_teachers):
            self.models[i] = self.models[i].from_pretrained("./model/teacher_"  + str(i))


    def aggregate(self, model_votes, batch_size):
        """Aggregate model output into a single tensor of votes of all models.
           Args:
                votes: Model output
                n_dataset: Number of datapoints
           Returns:
                counts: Torch tensor with counts across all models    
           """

        counts = torch.zeros([batch_size, 2])
        model_counts = torch.zeros([self.args.n_teachers, batch_size])
        model_index = 0

        for model in model_votes:
            index = 0
            for val in  model_votes[model]:
                i=0 if val[0]>val[1] else 1
                counts[index][i] += 1
                model_counts[model_index][index] = i
                index += 1

            model_index += 1

        return counts, model_counts

    def save_models(self,model,model_name):
        save_path = f'./model/{model_name}'  # 每个模型的保存路径

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存模型
        model.save_pretrained(save_path)

        print("\n")
        print("MODELS SAVED")
        print("\n")

    def analyze(self, preds, indices, moments=8):

        print(f'noise_eps:{self.epsilon}')
        datadepeps, dataindeps = pate.perform_analysis_torch(
            preds, indices, noise_eps=self.epsilon, delta=1e-6, moments=moments, beta=0.09
        )
        return datadepeps, dataindeps

    def predict(self, data,mask,outputs):
        """Make predictions using Noisy-max using Laplace mechanism.
           Args:
                data: Data for which predictions are to be made
           Returns:
                predictions: Predictions for the data
        """

        model_predictions = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data=data.to(device)
        mask=mask.to(device)
        with torch.no_grad():
            for i in range(self.n_teachers):
                model_predictions[i] = outputs[i]

        counts, model_counts = self.aggregate(model_predictions, len(data))
        predictions_nonoise = []
        for batch in counts:

            predictions_nonoise.append(torch.tensor(batch.max(dim=0)[1].long()).clone().detach())


        counts = counts.apply_(self.addnoise)
        predictions = []
        for batch in counts:

            predictions.append(torch.tensor(batch.max(dim=0)[1].long()).clone().detach())

        output = {"predictions": predictions,"predictions_nonoise":predictions_nonoise, "counts": counts, "model_counts": model_counts}

        return output
