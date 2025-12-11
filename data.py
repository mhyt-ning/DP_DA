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
        self.tokenizer = AutoTokenizer.from_pretrained('../model/roberta-large')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        
        encoding = self.tokenizer(text, max_length=512, truncation=True, padding='max_length')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_teacher_data(n_teachers, batch_size, target_epsilon):

    teacher_path = 'train_test/train_test_3.5.csv' 
    gen_path='train_test/eps'+str(target_epsilon)+'/gen_data_teacher.csv'

    data = pd.read_csv(teacher_path).sample(frac=1, random_state=42)
    train_data = data[:600]
    train_texts = train_data['input'].tolist()
    train_labels = train_data['output'].tolist()

    test_data = data[600:1000]
    test_texts = test_data['input'].tolist()
    test_labels = test_data['output'].tolist()

    gen_data = pd.read_csv(gen_path).sample(frac=1, random_state=42)
    gen_data = gen_data[:10000]
    gen_texts = gen_data['input'].tolist()
    gen_labels = gen_data['output'].tolist()


    print(f'{teacher_path}, {len(train_data)}')
    print(f'{teacher_path},{len(test_data)}')
    print(f'{gen_path},{len(gen_data)}')

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

    for i in range(len(gen_texts)):
        gen_texts[i] = str(gen_texts[i]).lower()
        if str(gen_labels[i]) == 'True':
            gen_labels[i] = 1
        else:
            gen_labels[i] = 0


    t_train_datasets = []
    t_test_datasets = []
    for i in range(n_teachers):
        train_len = len(train_texts) // n_teachers
        test_len = len(test_texts) // n_teachers
        print(f'no {i}: {train_len}, {test_len}')
        t_train_datasets.append(TextDataset(train_texts[i * train_len:(i + 1) * train_len],
                                            train_labels[i * train_len:(i + 1) * train_len]))
        t_test_datasets.append(
            TextDataset(test_texts[i * test_len:(i + 1) * test_len], test_labels[i * test_len:(i + 1) * test_len]))

    t_train_loader = []
    t_test_loader = []
    print("batch_size: ", (batch_size))
    for i in range(n_teachers):
        t_train_loader.append(DataLoader(t_train_datasets[i], batch_size, shuffle=True))
        t_test_loader.append(DataLoader(t_test_datasets[i], batch_size, shuffle=True))

    gen_dataset = TextDataset(gen_texts, gen_labels)
    gen_loader = DataLoader(gen_dataset, batch_size, shuffle=False)

    test_dataset = TextDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return t_train_loader, test_loader, gen_loader


class NoisyDataset(Dataset):
    """Dataset with targets predicted by ensemble of teachers.
       Args:
            dataloader (torch dataloader): The original torch dataloader.
            model(torch model): Teacher model to make predictions.
            transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, dataloader, predictionfn, transform=None):
        self.dataloader = dataloader
        self.predictionfn = predictionfn
        self.transform = transform
        # self.noisy_predict = noisy_predict
        self.noisy_data = self.process_data()

    def process_data(self):
        """
        Replaces original targets with targets predicted by ensemble of teachers.
        Returns:
            noisy_data[torch tensor]: Dataset with labels predicted by teachers

        """

        noisy_data = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_input_ids = []
        test_mask = []
        test_label = []
        i = 0
        for batch in self.dataloader:
            test_input_ids.append(batch['input_ids'])
            test_mask.append(batch['attention_mask'])
            test_label.append(batch['label'])
            i += 1
        print("noisy data complete")
        noisy_data = {'input_ids': test_input_ids, 'attention_mask': test_mask, "predictions": test_label}

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