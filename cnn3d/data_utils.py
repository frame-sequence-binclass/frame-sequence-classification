import os
import glob
import torch
import random
from PIL import Image
from typing import Tuple
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Assumes format: "videoID_frame_NUMBER.ext" or similar
def get_frame_number(file_path: str) -> int:
    filename = os.path.basename(file_path)
    parts = os.path.splitext(filename)[0].split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0 

class SequenceDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 5, step: int = 1, transform=None):
        super(SequenceDataset, self).__init__()
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.step = step
        self.transform = transform
       
        image_dir = os.path.join(self.root_dir, 'images')
        labels_dir = os.path.join(self.root_dir, 'labels')

        if not os.path.isdir(image_dir) or not os.path.isdir(labels_dir):
            print(f"Error: Missing 'images' or 'labels' directory in {root_dir}")
            self.sequences = []
            self.labels = []
            return

        all_image_files = sorted([
            f for f in glob.glob(os.path.join(image_dir, '*.jpg'))
            if not os.path.basename(f).startswith('.')
        ])
        if len(all_image_files) == 0:
            print(f"Warning: No images found in {image_dir}")
            self.sequences = []
            self.labels = []
            return
           
        # Group images by video source
        video_groups = {}
        for file_path in all_image_files:
            filename = os.path.basename(file_path)
            video_id = filename.split('_')[0]
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(file_path)
            
        self.sequences, self.labels = [], []
        positive_count, negative_count = 0, 0
        
        for video_id, image_files in video_groups.items():
            if len(image_files) < self.sequence_length:
                continue
            
            i = 0
            while i <= len(image_files) - self.sequence_length:
                # Check for gaps within the sequence starting at index i
                is_continuous = True

                for j in range(i, i + self.sequence_length - 1):
                    current_frame = get_frame_number(image_files[j])
                    next_frame = get_frame_number(image_files[j+1])
                
                    # Assume a "gap" if the next frame number is not exactly the current frame + 1
                    # You may adjust the gap tolerance
                    if next_frame != current_frame + 1:
                        is_continuous = False
                        break
                
                if is_continuous:
                    sequence_images = image_files[i:i + self.sequence_length]
                    cont = 0
                    sequence_label = 0
                    for img_path in sequence_images:
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        label_path = os.path.join(labels_dir, f"{img_name}.txt")
                        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                            cont += 1
                            
                            if cont >= 1:
                                sequence_label = 1
                                break
                            
                    self.sequences.append((sequence_images, sequence_label))
                    self.labels.append(sequence_label)
                    
                    if sequence_label == 1:
                        positive_count += 1
                    else:
                        negative_count += 1
                        
                    i += self.step
                else:
                    
                    print(f"Gap detected in video {video_id} between frames {get_frame_number(image_files[j])} and {get_frame_number(image_files[j+1])}. Skipping to next segment.")
                    i = j + 1 
        
        print(f"Found {positive_count} positive sequences.")
        print(f"Found {negative_count} negative sequences.")
           
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_paths, label = self.sequences[idx]
       
        image_tensors = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                image_tensors.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
       
        if not image_tensors:
            raise RuntimeError(f"Could not load any images for sequence at index {idx}")
           
        sequence_tensor = torch.stack(image_tensors, dim=0)
        return sequence_tensor, torch.tensor(label, dtype=torch.long)
    

def get_dataloader(path, batch_size=32, transform=None, sequence_len=5, step_size=1, shuffle=True):
    dataset = SequenceDataset(root_dir=path, sequence_length=sequence_len, step=step_size, transform=transform)
    class_counts = [dataset.labels.count(0), dataset.labels.count(1)]
    
    if shuffle:
        print(f"Number of samples in minority class: {min(class_counts[0], class_counts[1])}")
        num_samples = len(dataset)
        weights = [1.0 / count if count > 0 else 0 for count in class_counts]
        sample_weights = [weights[label] for label in dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  

    if len(dataloader) > 0:
        _, labels = next(iter(dataloader))
        label_counts = torch.bincount(labels)
        print(f"\n--- First Batch Class Distribution (Batch size: {batch_size}) ---")
        print(f"Number of samples in class 0 (Negative): {label_counts[0].item() if len(label_counts) > 0 and label_counts.shape[0] > 0 else 0}")
        print(f"Number of samples in class 1 (Positive): {label_counts[1].item() if len(label_counts) > 1 else 0}")
        print("------------------------------------------------------------\n")
    
    return dataloader