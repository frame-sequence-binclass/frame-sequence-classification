import os
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def test_model(test_loader, model):
    device = torch.device("cpu")
    model = model.to(device)
    y_pred, y_real = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predicted = torch.sigmoid(outputs)
            y_pred.extend(predicted.cpu().numpy())
            y_real.extend(labels.cpu().numpy())
    return y_real, y_pred

def results(y_real, y_predscore, thr, dataset_name, n):
    save_dir = "./frame_sequence/cnnlstm/results/"
    fpr, tpr, thresholds = roc_curve(y_real, y_predscore)
    auc_score = roc_auc_score(y_real, y_predscore)
    y_pred = [1 if score >= thr else 0 for score in y_predscore]
    cm = confusion_matrix(y_real , y_pred)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if not os.path.isdir(save_dir): os.mkdir(save_dir) 
    with open(save_dir+dataset_name+"_"+str(n)+".txt", "w") as f:
        f.write(f"ROC AUC: {auc_score:.4f}\n")
        print(f"ROC AUC: {auc_score:.4f}")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        print(f"Accuracy: {accuracy:.4f}")
        f.write(f"Precision: {precision:.4f}\n")
        print(f"Precision: {precision:.4f}")
        f.write(f"Recall: {recall:.4f}\n")
        print(f"Recall: {recall:.4f}")
        f.write(f"Confusion Matrix:\n{cm}\n")