import os
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def test_model(test_loader, model):
    device = torch.device('cpu')
    model  = model.to(device)
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            images = images.permute(0, 2, 1, 3, 4)
            outputs = model(images)
            predicted = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

def results(y_true, y_predscore, thr, video, n):
    save_dir = "./cnn3d/results/"
    fpr, tpr, thresholds = roc_curve(y_true, y_predscore)
    roc_auc = roc_auc_score(y_true, y_predscore)
    y_pred = [1 if score > thr else 0 for score in y_predscore]
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    with open(save_dir+video+"_"+str(n)+".txt", "w") as output:
        output.write(f"AUC: {roc_auc:.4f}" + "\n")
        print(f"AUC: {roc_auc:.4f}")
        output.write(f"Accuracy: {accuracy:.4f}" + "\n")
        print(f"Accuracy: {accuracy:.4f}")
        output.write(f"Precision: {precision:.4f}" + "\n")
        print(f"Precision: {precision:.4f}")
        output.write(f"Recall: {recall:.4f}" + "\n")
        print(f"Recall: {recall:.4f}")
        output.write(f"Confusion Matrix:\n{cm}\n")