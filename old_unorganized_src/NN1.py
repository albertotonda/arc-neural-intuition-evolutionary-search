import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class GridDataset(Dataset):
    def __init__(self, data, max_size=None):
        self.data = data
        if max_size is None:
            # For horizontal concatenation, we need to consider doubled width
            self.max_size = max(max(G1.shape[0], G2.shape[0], G1.shape[1], G2.shape[1]) for G1, G2, _ in data)
        else:
            self.max_size = max_size

    def pad_grid(self, grid):
        h, w = grid.shape
        pad_h = self.max_size - h
        pad_w = self.max_size - w
        padded = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
        return padded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        G1, G2, transformations = self.data[idx]
        G1_padded = self.pad_grid(G1)
        G2_padded = self.pad_grid(G2)
        return (
            torch.tensor(G1_padded, dtype=torch.float32).unsqueeze(0),
            torch.tensor(G2_padded, dtype=torch.float32).unsqueeze(0),
            torch.tensor(transformations, dtype=torch.float32)
        )

class EnhancedGridClassifier(nn.Module):
    def __init__(self, num_transformations):
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

def generate_structured_grid(size):
    grid = np.zeros((size, size))
    pattern_type = np.random.choice(['L_pattern', 'stairs', 'arrow', 'zigzag'])
    
    if pattern_type == 'L_pattern':
        grid[0:size-1, 0] = 10
        grid[size-1, 0:size//2] = 5
        grid[size//2, size//2] = 8
        grid[size//2, size//2 + 1] = 3
    
    elif pattern_type == 'stairs':
        for i in range(size//2):
            grid[size-1-i, i:i+2] = 10
            grid[size-1-i-2:size-1-i, i] = 5
        grid[0, size-1] = 8
        
    elif pattern_type == 'arrow':
        mid = size // 2
        grid[mid, :size-2] = 10
        grid[mid-1:mid+2, size-3:size] = np.array([
            [0, 5, 0],
            [10, 10, 10],
            [0, 5, 0]
        ])
        grid[mid-1:mid+2, 0:2] = 3
        
    elif pattern_type == 'zigzag':
        for i in range(0, size-2, 2):
            grid[i, i:i+3] = 10
            grid[i:i+2, i+2] = 5
        grid[0:2, size-2:size] = 8
    
    return grid

def generate_single_transform_data(num_samples=200, min_size=5, max_size=30):
    data = []
    for _ in range(num_samples):
        size = np.random.randint(min_size, max_size + 1)
        G1 = generate_structured_grid(size)
        
        transformations = np.zeros(6)
        transform_idx = np.random.randint(0, 6)
        transformations[transform_idx] = 1
        
        G2 = G1.copy()
        if transform_idx == 0:  # rot180
            G2 = np.rot90(G2, k=2)
        elif transform_idx == 1:  # vmirror
            G2 = np.flipud(G2)
        elif transform_idx == 2:  # hmirror
            G2 = np.fliplr(G2)
        elif transform_idx == 3:  # dmirror
            G2 = np.transpose(G2)[::-1]
        elif transform_idx == 4:  # hconcat
            G2 = np.hstack([G1, G1])
        elif transform_idx == 5:  # rot270
            G2 = np.rot90(G2, k=3)
            
        data.append((G1, G2, transformations))
    return data

def generate_multiple_transform_data(num_samples=200, min_size=5, max_size=30):
    data = []
    for _ in range(num_samples):
        size = np.random.randint(min_size, max_size + 1)
        G1 = generate_structured_grid(size)
        
        transformations = np.zeros(6)
        num_transforms = np.random.randint(2, 4)
        transform_indices = np.random.choice(6, num_transforms, replace=False)
        
        G2 = G1.copy()
        for idx in transform_indices:
            transformations[idx] = 1
            if idx == 0:  # rot180
                G2 = np.rot90(G2, k=2)
            elif idx == 1:  # vmirror
                G2 = np.flipud(G2)
            elif idx == 2:  # hmirror
                G2 = np.fliplr(G2)
            elif idx == 3:  # dmirror
                G2 = np.transpose(G2)[::-1]
            elif idx == 4:  # hconcat
                G2 = np.hstack([G2, G2])
            elif idx == 5:  # rot270
                G2 = np.rot90(G2, k=3)
                
        data.append((G1, G2, transformations))
    return data

def train_model(model, train_loader, test_loaders, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    best_test_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for G1, G2, labels in train_loader:
            G1, G2, labels = G1.to(device), G2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(G1, G2)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            test_losses = []
            for loader_name, test_loader in test_loaders.items():
                test_loss, test_accuracy, _ = evaluate_model(
                    model, test_loader, criterion, device, loader_name)
                test_losses.append(test_loss)
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    torch.save(model.state_dict(), 'best_model.pth')
            scheduler.step(sum(test_losses) / len(test_losses))

def evaluate_model(model, dataloader, criterion, device, dataset_name=""):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for G1, G2, labels in dataloader:
            G1, G2, labels = G1.to(device), G2.to(device), labels.to(device)
            outputs = model(G1, G2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = (all_predictions == all_labels).mean() * 100
    
    transformation_names = ['Rot180', 'VMirror', 'HMirror', 'DMirror', 'HConcat', 'Rot270']
    report = classification_report(all_labels, all_predictions, 
                                 target_names=transformation_names,
                                 zero_division=0)
    
    print(f"\nResults for {dataset_name}:")
    print(f"Loss: {total_loss / len(dataloader):.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nDetailed performance report:")
    print(report)
    
    return total_loss / len(dataloader), accuracy, report

if __name__ == "__main__":
    # Hyperparameters
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.0005
    NUM_TRANSFORMATIONS = 6  # Updated for new transformations
    BATCH_SIZE = 32
    MAX_GRID_SIZE = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate training data
    train_data = generate_single_transform_data(num_samples=2000) + \
                 generate_multiple_transform_data(num_samples=2000)
    
    # Generate test sets
    single_transform_test = generate_single_transform_data(num_samples=400)
    multiple_transform_test = generate_multiple_transform_data(num_samples=400)
    
    # Create datasets and dataloaders
    train_dataset = GridDataset(train_data, max_size=MAX_GRID_SIZE*2)  # *2 for hconcat
    single_test_dataset = GridDataset(single_transform_test, max_size=MAX_GRID_SIZE*2)
    multiple_test_dataset = GridDataset(multiple_transform_test, max_size=MAX_GRID_SIZE*2)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    single_test_loader = DataLoader(single_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    multiple_test_loader = DataLoader(multiple_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_loaders = {
        "Single Transform": single_test_loader,
        "Multiple Transform": multiple_test_loader
    }

    # Initialize model, criterion, optimizer and scheduler
    model = EnhancedGridClassifier(num_transformations=NUM_TRANSFORMATIONS)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Starting training...")
    train_model(model, train_loader, test_loaders, criterion, optimizer, scheduler, NUM_EPOCHS, device)
    
    print("\nFinal Evaluation:")
    print("=" * 50)
    
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, single_test_loader, criterion, device, "Single Transformation Test Set")
    evaluate_model(model, multiple_test_loader, criterion, device, "Multiple Transformations Test Set")