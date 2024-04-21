import os
import torchvision
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xception import xception
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax
import pandas as pd

model = xception(pretrained=False, num_classes=2)
model.load_state_dict(torch.load('/kaggle/input/xception_30epcohs/pytorch/xception_30epochs/1/xception_0.0001_32_30.pth'))
model.eval()

# Define the path to the folder
folder_path = "/kaggle/input/test-sets/test"

# Get all subdirectories in the folder
subdirectories = [f.path for f in os.scandir(folder_path) if f.is_dir()]

transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

results = []

for subdirectory in subdirectories:
    
    print(subdirectory)

    testset = torchvision.datasets.ImageFolder(root=subdirectory, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    num_images = len(testloader.dataset)
    gt_array = np.zeros(num_images)
    pred_array = np.zeros(num_images)
    raw_pred_array = np.zeros(num_images)

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    running_loss = 0.0
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model.to(device)

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
                
                
    softmax_array = softmax(raw_pred_array)
    fpr, tpr, _ = roc_curve(gt_array, softmax_array)
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
    plt.title(f"ROC {subdirectory.split('/')[-1]}")
    plt.legend(loc="lower right")
    # save the figure
    plt.savefig(f"ROC_curve_{subdirectory.split('/')[-1]}_30e.png")
    plt.show()
    
    
    method_name = subdirectory.split('/')[-1]  # Extract method name from directory
    results.append({"Method": method_name, "AUC": roc_auc, "AC": accuracy, "Loss": loss})
    
df = pd.DataFrame(results)
# Save the DataFrame to a CSV file
df.to_csv('Xception_results_32_0.0001_30.csv', index=False)