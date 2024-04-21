import torchvision
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import wandb
from xception import xception
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt



class train_model():
    def __init__(self, model, epochs, lr, batch_size, path):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.path = path
        self.trainloader, self.trainset, self.valloader, self.valset = self.load_data()

    def load_data(self):
        transform = transforms.Compose(
            [transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.ImageFolder(root=self.path + 'train', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)

        valset = torchvision.datasets.ImageFolder(root=self.path + 'val', transform=transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=2)
        
        #testset = torchvision.datasets.ImageFolder(root=self.path + 'test', transform=transform)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
        #                                         shuffle=False, num_workers=2)

        return trainloader, trainset, valloader, valset#, testloader, testset
    


    def test_model(self, data_loader, set = "val"):
        criterion = nn.CrossEntropyLoss()

        num_images = len(data_loader.dataset)
        gt_array = np.zeros(num_images)
        pred_array = np.zeros(num_images)

        correct = 0
        total = 0
        running_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                gt_array[i*labels.size(0):(i+1)*labels.size(0)] = labels.cpu().numpy()
                pred_array[i*labels.size(0):(i+1)*labels.size(0)] = predicted.cpu().numpy()


        accuracy = correct / total * 100
        loss = running_loss/total


        print("Loss on val set:", loss)
        print("Accuracy on val set:", accuracy)
    
        return gt_array, pred_array, loss, accuracy

    def train(self):
        self.model.cuda()

        
        wandb.init(
           project = "Xception_AdamW",
            name = f"lr_{self.lr}_epochs_{self.epochs}_batch_size_{self.batch_size}_platoLRS",
            config ={
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optimizer": "AdamW"
            }
        )

        #define the loss function
        criterion = nn.CrossEntropyLoss()

        #optimization parameters: lr=0.045, momentum = 0.9, learning rate decay = 0.94 every 2 epochs

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        #learning rate scheduler

        #scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience =  5, verbose=True)
        

        for epoch in range(self.epochs):
            with tqdm(total=len(self.trainset), desc = "Epoch: " + str(epoch) + "/" + str(self.epochs), unit='img') as prog_bar:
                for i, data in enumerate(self.trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    prog_bar.set_postfix(**{'Loss': loss.data.cpu().detach().numpy()})
                    prog_bar.update(self.batch_size)

            #if epoch % 10 == 0:
            _, _, val_loss, val_accuracy = self.test_model(self.valloader, set = "val")

            #scheduler.step()
            scheduler.step(val_loss)

            wandb.log({"Val_Loss": val_loss, "Val_Accuracy": val_accuracy, "epoch": epoch})

        wandb.finish()

        return self.model
    
    



    

if __name__ == '__main__':
    root_path = '/kaggle/input/f-deppfake-trainandval/assignment/'
    lr = 0.0001
    epochs = 30
    batch_size = 32
    model = xception(pretrained=False, num_classes=2)
    trainer = train_model(model, epochs, lr, batch_size, root_path)
    trained_model = trainer.train()
    torch.save(trained_model.state_dict(), f'xception_{lr}_{batch_size}_{epochs}_platoLRS.pth')






    


    