import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import os
from tqdm import tqdm


class Student:
    """Implementation of Student models
       The student model is trained from the public data labelled by teacher ensembles.
       The teacher ensembles were trained using sensitive data. The student model is further
       used to make predictions on public data.
       Args:
           args[Arguments obj]: Object of arguments class used to control hyperparameters
           model[torch model]: Model of Student 
    """

    def __init__(self, args):

        self.args = args
        self.model = RobertaForSequenceClassification.from_pretrained('../model/roberta-large',
                                                             num_labels=2)
                                                             

    def predict(self, data,mask):
        """Function which accepts unlabelled public data and labels it using 
           teacher's model.
           Args:
               model[torch model]: Teachers model
               data [torch tensor]: Public unlabelled data
           Returns:
               dataset[Torch tensor]: Labelled public dataset
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data=data.to(device)
        mask=mask.to(device)
        output=self.model(data,mask).logits.cpu()
        # print(output)
        re=torch.max(output, 1)[1]
        # print(re)
        return output

    def train(self, dataset):
        """Function to train the student model.
           Args:
               dataset[torch dataset]: Dataset using which model is trained.
        """

        for epoch in range(0, self.args.student_epochs):
            self.loop_roberta(dataset, epoch)
        self.save_models(self.model,'student')

    def loop_roberta(self,data_loader,epoch):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader=data_loader
        model=self.model
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)

        model.train()
        for i in range(len(train_loader)):
            input_ids = train_loader['input_ids'][i].to(device)
            attention_mask = train_loader['attention_mask'][i].to(device)
            labels = train_loader['predictions'][i].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(f"Epoch {epoch + 1}/{self.args.student_epochs}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

    def save_model(self):
        torch.save(self.model.state_dict(), "Models/" + "student_model")

    def save_models(self,model,model_name):
        save_path = f'{self.args.save_path}/{model_name}' 

    
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存模型
        model.save_pretrained(save_path)

        print("\n")
        print("MODELS SAVED")
        print("\n")
    
    def load_models(self):
    
        path_name = "model_"
    
        self.model = self.model.from_pretrained(f"{self.args.save_path}/student")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)  

