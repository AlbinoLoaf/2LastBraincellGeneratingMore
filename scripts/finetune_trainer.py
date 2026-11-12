import os
import numpy as np
import torch
import json
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

class Trainer(object):
    def __init__(self, params, data_loaders, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
           # visible_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", self.params.cuda))
            torch.cuda.set_device(0)#(visible_device)
            
            self.device = torch.device(f"cuda:0")
            print(f"--- GPU ENGAGED: Using {torch.cuda.get_device_name(0)} (Device ID: {0}) ---")
        else:
            self.device = torch.device("cpu")
            print("--- WARNING: CUDA NOT AVAILABLE. RUNNING ON CPU ---")

        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders['val']
        self.test_loader = data_loaders['   test']
        
        self.model = model.to(self.device)
        self.data_length = len(self.train_loader)
        
        label_smoothing_val = getattr(self.params, 'label_smoothing', 0.1)
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing_val).to(self.device)

        if getattr(self.params, 'parallel', False):
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        head_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'proj_out' in name: 
                head_params.append(param)
            else:
                backbone_params.append(param)

        head_lr = 0.001 * ((self.params.batch_size / 256) ** 0.5)
        
        optim_groups = [
            {'params': backbone_params, 'lr': self.params.lr},
            {'params': head_params, 'lr': head_lr}
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups, 
            weight_decay=self.params.weight_decay
        )

        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.params.epochs * self.data_length, 
            eta_min=1e-6
        )

    def train(self):
        best_val_acc = 0.0
        best_model_path = rf'{self.params.model_dir}/best_model.pth'
        os.makedirs(self.params.model_dir, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_kappa': [],
            'val_f1': []
        }
        
        for epoch in range(self.params.epochs):
            self.model.train()
            train_losses = []
            
            for x, labels in tqdm(self.train_loader, mininterval=10, desc=f"Epoch {epoch+1} Training"):
                self.optimizer.zero_grad()
                
                x = x.float().to(self.device)
                labels = labels.long().to(self.device)
                
                batch_size = x.shape[0]
                x = x.view(batch_size, 32, 10, 200)
                
                outputs = self.model(x)
                if outputs.dim() == 4:
                    outputs = outputs.mean(dim=(1, 2))
                    
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                
                self.optimizer.step()
                self.optimizer_scheduler.step()
                train_losses.append(loss.item())
                
            mean_train_loss = np.mean(train_losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr'] 
            
            val_loss, val_acc, val_kappa, val_f1 = self.evaluate(self.val_loader)
            
            history['train_loss'].append(mean_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_kappa'].append(val_kappa)
            history['val_f1'].append(val_f1)
            
            with open(os.path.join(self.params.model_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=4)

            print(f'Epoch {epoch+1}: Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'Val Bal-Acc: {val_acc:.4f} | Val Kappa: {val_kappa:.4f} | Val F1: {val_f1:.4f} | LR: {learning_rate:.6f}')
            
            if val_acc > best_val_acc:
                peak_model_path = rf'{self.params.model_dir}/epoch{epoch+1}_valacc{val_acc:.4f}.pth'
                torch.save(self.model.state_dict(), peak_model_path)
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved in {peak_model_path}")
                best_val_acc = val_acc


    def evaluate(self, data_loader):
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, labels in data_loader:
                x = x.float().to(self.device)
                labels = labels.long().to(self.device)
                
                batch_size = x.shape[0]
                x = x.view(batch_size, 32, 10, 200)
                
                outputs = self.model(x)
                if outputs.dim() == 4:
                    outputs = outputs.mean(dim=(1, 2))
                    
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        mean_loss = np.mean(losses)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return mean_loss, bal_acc, kappa, f1
