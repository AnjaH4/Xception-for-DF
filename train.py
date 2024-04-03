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
        
        # testset = torchvision.datasets.ImageFolder(root=self.path + 'test', transform=transform)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
        #                                         shuffle=False, num_workers=2)

        return trainloader, trainset, valloader, valset #, testloader, testset
    


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

        #wandb init
        wandb.init(
            project = "Xception",
            name = f"lr_{self.lr}_epochs_{self.epochs}_batch_size_{self.batch_size}",
            config ={
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size
            }
        )

        #define the loss function
        criterion = nn.CrossEntropyLoss()

        #optimization parameters: lr=0.045, momentum = 0.9, learning rate decay = 0.94 every 2 epochs

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        #learning rate scheduler

        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)

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

            scheduler.step()

            wandb.log({"Val_Loss": val_loss, "Val_Accuracy": val_accuracy})

        wandb.finish()

        return self.model
    
# def final_test (model, path, name):
#     transform = transforms.Compose(
#         [transforms.Resize((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     testset = torchvision.datasets.ImageFolder(root=path, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32,
#                                             shuffle=False, num_workers=2)

#     criterion = nn.CrossEntropyLoss()

#     num_images = len(testloader.dataset)
#     gt_array = np.zeros(num_images)
#     pred_array = np.zeros(num_images)

#     correct = 0
#     total = 0
#     running_loss = 0.0
#     model.eval()

#     with torch.no_grad():
#         for i, data in enumerate(testloader):
#             inputs, labels = data
#             inputs, labels = inputs.cuda(), labels.cuda()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             gt_array[i*labels.size(0):(i+1)*labels.size(0)] = labels.cpu().numpy()
#             pred_array[i*labels.size(0):(i+1)*labels.size(0)] = predicted.cpu().numpy()

    

#     #make a ROC curve
#     fpr, tpr, _ = roc_curve(gt_array, pred_array)
#     roc_auc = auc(fpr, tpr)

#     print("AUC:", roc_auc)
#     print("Accuracy:", accuracy)
#     print("Loss:", loss)

#     #plot the ROC curve
#     sns.set_theme()
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC')
#     plt.legend(loc="lower right")
#     #save the figure
#     plt.savefig(f"ROC_curve_{name}.png")
#     plt.show()

def final_test(model, path, name):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    num_images = len(testloader.dataset)
    gt_array = np.zeros(num_images)
    pred_array = np.zeros(num_images)

    correct = 0
    total = 0
    running_loss = 0.0
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    raw_pred_array = np.zeros(num_images)
    
    with torch.no_grad(), tqdm(total=len(testloader)) as pbar:
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            gt_array[i*labels.size(0):(i+1)*labels.size(0)] = labels.cpu().numpy()
            pred_array[i*labels.size(0):(i+1)*labels.size(0)] = predicted.cpu().numpy()
            raw_pred_array[i*labels.size(0):(i+1)*labels.size(0)] = outputs[:, 1].cpu().numpy()

            pbar.update(1)

    # make a ROC curve
    fpr, tpr, _ = roc_curve(gt_array, pred_array)
    roc_auc = auc(fpr, tpr)

    print("AUC:", roc_auc)
    accuracy = correct / total
    print("Accuracy:", accuracy)
    print("Loss:", running_loss / len(testloader))

    # plot the ROC curve
    sns.set_theme()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # save the figure
    plt.savefig(f"ROC_curve_{name}.png")
    plt.show()

# Example usage:
# final_test(model, "FF+_align/assignment/test/test_deep_fakes", "DeepFakes")

    

if __name__ == '__main__':
    root_path = 'FF+_align/assignment/'
    lr = 0.0002
    epochs = 3
    batch_size = 4
    model = xception(pretrained=False, num_classes=2)
    trainer = train_model(model, 3, 0.045, 4, root_path)
    trained_model = trainer.train()
    gt_array, pred_array, loss, accuracy = trainer.test_model(trainer.testloader, set = "test")
    print("Loss on test set:", loss)
    print("Accuracy on test set:", accuracy) 
    #save the model
    torch.save(trained_model.state_dict(), f'xception_{lr}_{batch_size}_{epochs}.pth')






    


    