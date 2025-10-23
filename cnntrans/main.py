import os
from torchvision import transforms
from data_utils import get_dataloader
from train import train_model, load_trained_model
from test import test_model, results

def main():
    ROOT_DIR   = "/dataset"
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    N_SEQUENCE = 5
    STEP_SIZE  = 2
    TRANSFORMS = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = get_dataloader(path = os.path.join(ROOT_DIR, "train"), 
                                  batch_size = BATCH_SIZE,
                                  transform = TRANSFORMS,
                                  step_size = STEP_SIZE, 
                                  sequence_len = N_SEQUENCE)
    val_loader   = get_dataloader(path = os.path.join(ROOT_DIR, "val"), 
                                  batch_size = BATCH_SIZE,
                                  transform = TRANSFORMS,
                                  step_size = STEP_SIZE, 
                                  sequence_len = N_SEQUENCE,
                                  shuffle=False)
    test_loader  = get_dataloader(path = os.path.join(ROOT_DIR, "test"), 
                                  batch_size = BATCH_SIZE,
                                  transform = TRANSFORMS,
                                  step_size = STEP_SIZE, 
                                  sequence_len = N_SEQUENCE,
                                  shuffle=False)

    model = train_model(train_loader, val_loader, IMAGE_SIZE, N_SEQUENCE, pretrained = True)
    #model = load_trained_model(IMAGE_SIZE, N_SEQUENCE, pretrained = True)

    n = 0
    y_real, y_pred = test_model(test_loader, model)
    for thr in [0.25, 0.5, 0.75]:
        results(y_real, y_pred, thr, "S", n)
        n+=1


if __name__ == '__main__':
    main()