import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from sklearn.metrics import classification_report

class ARCGridDataset(Dataset):
    def __init__(self, input_grids, output_grids, task_ids, max_size=30):
        self.data = list(zip(input_grids, output_grids, task_ids))
        self.max_size = max_size
        self.transform_map = {
            "3c9b0459": 0,  # rot180
            "6150a2bd": 0,  # rot180 (same index)
            "67a3c6ac": 1,  # vmirror
            "68b16354": 2,  # hmirror
            "74dd1130": 3,  # dmirror
            "a416b8f3": 4,  # hconcat
            "ed36ccf7": 5   # rot270
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_grid, output_grid, task_id = self.data[idx]
        transform_label = np.zeros(6)
        transform_label[self.transform_map[task_id]] = 1
        return (
            torch.tensor(input_grid, dtype=torch.float32),
            torch.tensor(output_grid, dtype=torch.float32),
            torch.tensor(transform_label, dtype=torch.float32)
        )

def get_class_weights(task_ids):
    transform_counts = {i: 0 for i in range(6)}
    transform_map = {
        "3c9b0459": 0, "6150a2bd": 0,  # both map to rot180
        "67a3c6ac": 1, "68b16354": 2,
        "74dd1130": 3, "a416b8f3": 4,
        "ed36ccf7": 5
    }
    
    for task_id in task_ids:
        transform_counts[transform_map[task_id]] += 1
    
    total_samples = sum(transform_counts.values())
    class_weights = [total_samples / (6 * count) if count > 0 else 1.0 
                    for count in transform_counts.values()]
    
    return torch.FloatTensor(class_weights)

class EnhancedGridClassifier(nn.Module):
    def __init__(self, num_transformations=6):
        super(EnhancedGridClassifier, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_transformations),
            nn.Sigmoid()
        )

    def forward(self, G1, G2):
        x = torch.cat((G1, G2), dim=1)
        att = self.attention(x)
        x = x * att
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_and_preprocess_arc_tasks(task_files, max_size=30):
    all_input_grids = []
    all_output_grids = []
    all_task_ids = []
    
    for task_file in task_files:
        task_id = os.path.splitext(os.path.basename(task_file))[0]
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        for pair in task_data:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            padded_input = -np.ones((max_size, max_size))
            padded_output = -np.ones((max_size, max_size))
            
            start_x = (max_size - input_grid.shape[0]) // 2
            start_y = (max_size - input_grid.shape[1]) // 2
            padded_input[start_x:start_x + input_grid.shape[0], 
                        start_y:start_y + input_grid.shape[1]] = input_grid
            
            start_x = (max_size - output_grid.shape[0]) // 2
            start_y = (max_size - output_grid.shape[1]) // 2
            padded_output[start_x:start_x + output_grid.shape[0], 
                         start_y:start_y + output_grid.shape[1]] = output_grid
            
            padded_input = (padded_input + 1) / 10
            padded_output = (padded_output + 1) / 10
            
            padded_input = padded_input[np.newaxis, ...]
            padded_output = padded_output[np.newaxis, ...]
            
            all_input_grids.append(padded_input)
            all_output_grids.append(padded_output)
            all_task_ids.append(task_id)
    
    return np.array(all_input_grids), np.array(all_output_grids), all_task_ids

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for input_grids, output_grids, labels in dataloader:
            input_grids = input_grids.to(device)
            output_grids = output_grids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_grids, output_grids)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = (all_predictions == all_labels).mean() * 100
    avg_loss = running_loss / len(dataloader)
    
    transformation_names = ['rot180', 'vmirror', 'hmirror', 'dmirror', 'hconcat', 'rot270']
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_predictions, 
        target_names=transformation_names,
        zero_division=0
    ))
    
    # Print per-class metrics
    class_accuracies = []
    for i in range(6):
        class_mask = all_labels[:, i] == 1
        if np.any(class_mask):
            class_acc = (all_predictions[class_mask, i] == all_labels[class_mask, i]).mean() * 100
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    print("\nPer-transformation Accuracies:")
    for name, acc in zip(transformation_names, class_accuracies):
        print(f"{name:>8}: {acc:.2f}%")
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('Training:', end=' ')
        
        for batch_idx, (input_grids, output_grids, labels) in enumerate(train_loader):
            input_grids = input_grids.to(device)
            output_grids = output_grids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_grids, output_grids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print('.', end='', flush=True)
        
        train_loss = running_loss / len(train_loader)
        print(f'\nTraining Loss: {train_loss:.4f}')
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved! (Validation Accuracy: {val_accuracy:.2f}%)')

if __name__ == "__main__":
    # Setup parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.0005
    MAX_GRID_SIZE = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    task_ids = ["3c9b0459", "6150a2bd", "67a3c6ac", "68b16354", "74dd1130", "a416b8f3", "ed36ccf7"]
    task_files = [f"shapes_dataset/tasks/{task_id}.json" for task_id in task_ids]
    input_grids, output_grids, task_ids = load_and_preprocess_arc_tasks(task_files)
    
    # Calculate class weights
    class_weights = get_class_weights(task_ids).to(device)
    print("\nClass weights:", class_weights.cpu().numpy())
    
    # Split data
    total_samples = len(input_grids)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = ARCGridDataset(
        input_grids[train_indices], 
        output_grids[train_indices], 
        [task_ids[i] for i in train_indices]
    )
    val_dataset = ARCGridDataset(
        input_grids[val_indices], 
        output_grids[val_indices], 
        [task_ids[i] for i in val_indices]
    )
    test_dataset = ARCGridDataset(
        input_grids[test_indices], 
        output_grids[test_indices], 
        [task_ids[i] for i in test_indices]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model, criterion with weights, and optimizer
    model = EnhancedGridClassifier()
    criterion = nn.BCELoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Train model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("=" * 50)
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")