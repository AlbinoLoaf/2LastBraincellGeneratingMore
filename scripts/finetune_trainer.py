import os
import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

class Trainer(object):
    def __init__(self, params, data_loaders, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        
        # Unpack the dictionary of dataloaders
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders['val']
        self.test_loader = data_loaders['test']
        
        self.model = model.to(self.device)
        
        # Classification loss instead of MSE
        self.criterion = CrossEntropyLoss().to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.train_loader)

        summary(self.model, input_size=(1, 32, 10, 200))
        
        # Fixed hardcoded profiler dimensions for 32 channels
        macs, model_params = get_model_complexity_info(self.model, (32, 10, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', model_params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        if self.params.lr_scheduler == 'CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler == 'StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        # Fallback scheduler
        else:
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )

    def train(self):
        best_val_acc = 0.0
        os.makedirs(self.params.model_dir, exist_ok=True)
        
        for epoch in range(self.params.epochs):
            self.model.train()
            train_losses = []
            
            # Capture both x and labels for classification
            for x, labels in tqdm(self.train_loader, mininterval=10, desc=f"Epoch {epoch+1} Training"):
                self.optimizer.zero_grad()
                
                x = x.float().to(self.device) / 100.0
                labels = labels.long().to(self.device)
                
                batch_size = x.shape[0]
                x = x.view(batch_size, 32, 10, 200)
                
                # Model outputs logits for classification
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
            
            # Run validation phase
            val_loss, val_acc, val_kappa, val_f1 = self.evaluate(self.val_loader)
            
            print(f'Epoch {epoch+1}: Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'Val Bal-Acc: {val_acc:.4f} | Val Kappa: {val_kappa:.4f} | Val F1: {val_f1:.4f} | LR: {learning_rate:.6f}')
            
            # Save based on validation accuracy
            if val_acc > best_val_acc:
                peak_model_path = rf'{self.params.model_dir}/epoch{epoch+1}_valacc{val_acc:.4f}.pth'
                torch.save(self.model.state_dict(), peak_model_path)
                best_model_path = rf'{self.params.model_dir}/best_model.pth'
                torch.save(self.model.state_dict(), best_model_path)
                
                print(f"New best model saved in {peak_model_path}")
                best_val_acc = val_acc
        # Final test evaluation after training completes
        print("\n--- Final Test Evaluation ---")
        self.model.load_state_dict(torch.load(rf'{self.params.model_dir}/best_model.pth'))
        test_loss, test_acc, test_kappa, test_f1 = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f} | Test Bal-Acc: {test_acc:.4f} | Test Kappa: {test_kappa:.4f} | Test F1: {test_f1:.4f}')


    def evaluate(self, data_loader):
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, labels in data_loader:
                x = x.float().to(self.device) / 100.0
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
