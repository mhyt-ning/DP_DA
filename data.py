import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokenizer = AutoTokenizer.from_pretrained('../model/roberta-large')
        encoding = tokenizer(text, max_length=512, truncation=True, padding='max_length')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(n_teachers,batch_size):
    """Helper function used to load the train/test data.
       Args:
           train[boolean]: Indicates whether its train/test data.
           batch_size[int]: Batch size
    """
    train_path='train_test/train_test_3.5.csv'
    test_path='train_test/train_test_3.5.csv'
    data=pd.read_csv(train_path).sample(frac=1, random_state=42)
    # train_data = pd.read_csv(train_path).sample(frac=1, random_state=42)[:600]
    train_data=data[:600]
    # train_texts = train_data['input'][:num].tolist()
    # train_labels = train_data['output'][:num].tolist()
    train_texts = train_data['input'].tolist()
    train_labels = train_data['output'].tolist()

    # test_data = pd.read_csv(test_path).sample(frac=1, random_state=42)[:400]
    test_data=data[600:]
    test_texts = test_data['input'].tolist()
    test_labels = test_data['output'].tolist()

    print(f'训练数据集：{train_path}，大小：{len(train_data)}')
    print(f'测试数据集：{test_path}，大小：{len(test_data)}')

    for i in range(len(train_texts)):
        train_texts[i] = str(train_texts[i]).lower()
        if str(train_labels[i]) == 'True':
            train_labels[i] = 1
        else:
            train_labels[i] = 0

    for i in range(len(test_texts)):
        test_texts[i] = str(test_texts[i]).lower()
        if str(test_labels[i]) == 'True':
            test_labels[i] = 1
        else:
            test_labels[i] = 0


    train_datasets=[]
    test_datasets=[]
    for i in range(n_teachers):
        train_len=len(train_texts)//n_teachers
        test_len=len(test_texts)//n_teachers
        print(f'第{i}个教师，训练集长度：{train_len}，测试集长度：{test_len}')
        train_datasets.append(TextDataset(train_texts[i*train_len:(i+1)*train_len],train_labels[i*train_len:(i+1)*train_len]))
        test_datasets.append(TextDataset(test_texts[i*test_len:(i+1)*test_len],test_labels[i*test_len:(i+1)*test_len]))

    train_loaders=[]
    test_loaders=[]
    print("batch_size: ",(batch_size))
    for i in range(n_teachers):
        train_loaders.append(DataLoader(train_datasets[i], batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_datasets[i], batch_size, shuffle=True))
    test_dataset = TextDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loaders, test_loaders,test_loader



class NoisyDataset(Dataset):
    """Dataset with targets predicted by ensemble of teachers.
       Args:
            dataloader (torch dataloader): The original torch dataloader.
            model(torch model): Teacher model to make predictions.
            transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, dataloader, predictionfn, noisy_predict,transform=None):
        self.dataloader = dataloader
        self.predictionfn = predictionfn
        self.transform = transform
        self.noisy_predict=noisy_predict
        self.noisy_data = self.process_data()

    def process_data(self):
        """
        Replaces original targets with targets predicted by ensemble of teachers.
        Returns:
            noisy_data[torch tensor]: Dataset with labels predicted by teachers
            
        """

        noisy_data = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_input_ids=[]
        test_mask=[]
        test_label=[]
        i=0
        for batch in self.dataloader:
            test_input_ids.append(batch['input_ids'])
            test_mask.append(batch['attention_mask'])
            test_label.append(torch.tensor(self.noisy_predict[i]))
            i+=1
        print("noisy data complete")
        noisy_data={'input_ids':test_input_ids, 'attention_mask':test_mask,"predictions":test_label}

        return noisy_data

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):

        sample = self.noisy_data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    load_data(2)

if __name__ == '__main__':
    main()