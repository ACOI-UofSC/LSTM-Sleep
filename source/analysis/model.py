import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import trange

from source.analysis.dataset import collate_fn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LocalGlobalLSTM(nn.Module):
    def __init__(self, feature_dim=4, local_hidden_dim=128, global_hidden_dim=256, local_steps=15, dropout=0.1,
                 n_class=2):
        super(LocalGlobalLSTM, self).__init__()
        self.local_lstm = nn.LSTM(feature_dim, local_hidden_dim, batch_first=True, bidirectional=True)

        self.local_fc = nn.Sequential(
            nn.Linear(local_hidden_dim * local_steps * 2, local_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden_dim, local_hidden_dim)
        )

        self.global_lstm = nn.LSTM(local_hidden_dim, global_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(global_hidden_dim * 2, n_class)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, lengths):
        batch_size, time_steps, num_channels, feature_dim = x.size()
        x = x.view(batch_size * time_steps, num_channels, feature_dim)
        local_lstm_out, _ = self.local_lstm(x)

        local_lstm_out = self.local_fc(local_lstm_out.reshape(batch_size * time_steps, -1))
        local_lstm_out = local_lstm_out.view(batch_size, time_steps, -1)

        packed_input = pack_padded_sequence(local_lstm_out, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.global_lstm(packed_input)
        global_lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(global_lstm_out)
        output = self.softmax(output)
        return output


class Trainer:
    def __init__(self, model, num_epochs=300, class_weight=None, device='cpu'):
        self.batch_size = 5
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.class_weight = class_weight.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        self.num_epochs = num_epochs
        self.val_score = 0
        self.best_model = None
        self.device = device
        self.collate_fn = collate_fn

    def set_train_data(self, dataset):
        self.train_set = dataset
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=self.collate_fn)

    def set_val_data(self, dataset):
        self.val_set = dataset
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=self.collate_fn)

    def set_test_data(self, dataset):
        self.test_set = dataset
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.collate_fn)

    def fit(self):
        self.model = self.model.to(self.device)
        for epoch in trange(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels, lengths in self.train_loader:
                inputs, labels = inputs.to(self.device).type(torch.float32), labels.to(self.device).long()
                self.optimizer.zero_grad()
                outputs = self.model(inputs, lengths)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                mask = (labels != -1)  # 掩盖填充部分
                outputs = outputs[mask]
                labels = labels[mask]
                loss = self.criterion(outputs, labels)

                # 加权损失
                weights = self.class_weight[labels]
                loss = loss * weights
                loss = loss.mean()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                break

            epoch_loss = running_loss / len(self.train_set)
            val_acc = self.eval()
            # self.test()
            print(f'Epoch {epoch + 1}/{self.num_epochs} Loss: {epoch_loss:.4f}  Val ACC:{val_acc}')

            if val_acc > self.val_score:
                self.val_score = val_acc
                self.best_model = self.model.state_dict()
        print('Training complete')
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        return self.model

    def eval(self):
        prediction, gt = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, lengths in self.val_loader:
                inputs = inputs.to(self.device).type(torch.float32)
                outputs = self.model(inputs, lengths)

                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                mask = (labels != -1)  # 掩盖填充部分
                outputs = outputs[mask]
                labels = labels[mask].cpu().numpy()

                p = np.argmax(outputs.cpu().numpy(), -1)
                splits = [p[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
                splits_labels = [labels[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]

                prediction.append(splits)
                gt.append(splits_labels)
        prediction = sum(prediction, [])
        prediction_np = np.concatenate(prediction)
        gt = sum(gt, [])
        gt_np = np.concatenate(gt)
        return sum(prediction_np == gt_np) / len(prediction_np)

    def test(self):
        score, prediction, gt = [], [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, lengths in self.test_loader:
                inputs = inputs.to(self.device).type(torch.float32)
                outputs = self.model(inputs, lengths)

                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                mask = (labels != -1)  # 掩盖填充部分
                outputs = outputs[mask]
                labels = labels[mask].cpu().numpy()

                p = outputs.cpu().numpy()
                p_score = np.max(p, -1)
                p_score_all = outputs.cpu().numpy()
                p = np.argmax(p, -1)
                splits = [p[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
                splits_score = [p_score_all[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
                splits_labels = [labels[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]

                prediction.append(splits)
                gt.append(splits_labels)
                score.append(splits_score)

        prediction = sum(prediction, [])
        gt = sum(gt, [])
        score = sum(score, [])

        return score, prediction, gt
