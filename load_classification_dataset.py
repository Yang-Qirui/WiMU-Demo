import torch
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_classification_dataset(data_dir, split='train'):
    """
    Load the classification dataset.
    
    Args:
        data_dir (str): Directory containing the classification data
        split (str): 'train' or 'test'
        
    Returns:
        tuple: (features, labels) as torch tensors
    """
    features_path = os.path.join(data_dir, f'{split}_features.pt')
    labels_path = os.path.join(data_dir, f'{split}_labels.pt')
    
    if os.path.exists(features_path) and os.path.exists(labels_path):
        features = torch.load(features_path)
        labels = torch.load(labels_path)
        return features, labels
    else:
        # Fallback to numpy arrays
        features = torch.from_numpy(np.load(os.path.join(data_dir, f'{split}_features.npy')))
        labels = torch.from_numpy(np.load(os.path.join(data_dir, f'{split}_labels.npy')))
        return features, labels

def load_coordinates(data_dir, split='train'):
    """
    Load coordinate data for the classification dataset.
    
    Args:
        data_dir (str): Directory containing the classification data
        split (str): 'train' or 'test'
        
    Returns:
        torch.Tensor: Coordinates as tensor of shape [N, 2]
    """
    coordinates_path = os.path.join(data_dir, 'coordinates.npy')
    if os.path.exists(coordinates_path):
        coordinates = torch.from_numpy(np.load(coordinates_path))
        return coordinates
    else:
        logger.warning(f"Coordinates file not found: {coordinates_path}")
        return torch.tensor([[0.0, 0.0]])  # Fallback

def load_dataset_stats(data_dir, split='train'):
    """
    Load dataset statistics.
    
    Args:
        data_dir (str): Directory containing the classification data
        split (str): 'train' or 'test'
        
    Returns:
        dict: Dataset statistics
    """
    stats_path = os.path.join(data_dir.replace('test_classification', 'train_classification'), 'dataset_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {}

def simple_classification_example():
    """
    Simple example of using the classification dataset with scikit-learn.
    """
    print("Loading classification dataset...")
    
    # Load training data
    train_features, train_labels = load_classification_dataset('data/train_classification', 'train')
    test_features, test_labels = load_classification_dataset('data/test_classification', 'test')
    
    # Load coordinates
    train_coordinates = load_coordinates('data/train_classification', 'train')
    test_coordinates = load_coordinates('data/test_classification', 'test')
    
    # Load dataset statistics
    train_stats = load_dataset_stats('data/train_classification')
    print(f"Dataset statistics: {train_stats}")
    
    # Convert to numpy for scikit-learn
    X_train = train_features.numpy()
    y_train = train_labels.numpy()
    X_test = test_features.numpy()
    y_test = test_labels.numpy()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Training coordinates shape: {train_coordinates.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Test coordinates shape: {test_coordinates.shape}")
    print(f"Number of unique labels: {len(np.unique(y_train))}")
    print(f"Label range: {np.min(y_train)} to {np.max(y_train)}")
    
    # Print coordinate statistics
    print(f"Training coordinate range: x=[{train_coordinates[:, 0].min():.2f}, {train_coordinates[:, 0].max():.2f}], y=[{train_coordinates[:, 1].min():.2f}, {train_coordinates[:, 1].max():.2f}]")
    
    # Train a simple Random Forest classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    feature_importance = clf.feature_importances_
    plt.figure(figsize=(12, 6))
    plt.plot(feature_importance)
    plt.title('Feature Importance (AP Unions)')
    plt.xlabel('AP Union Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature importance plot to feature_importance.png")
    
    # Plot waypoints by label
    if len(train_coordinates) > 0:
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(y_train)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = y_train == label
            coords = train_coordinates[mask]
            plt.scatter(coords[:, 0], coords[:, 1], c=[colors[i]], label=f'Waypoint {label}', alpha=0.7)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Waypoint Locations by Classification Label')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('waypoint_locations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved waypoint locations plot to waypoint_locations.png")
    
    return clf, accuracy, train_coordinates, test_coordinates

def pytorch_classification_example():
    """
    Example using PyTorch neural network for classification.
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    print("\nPyTorch Neural Network Example...")
    
    # Load data
    train_features, train_labels = load_classification_dataset('data/train_classification', 'train')
    test_features, test_labels = load_classification_dataset('data/test_classification', 'test')
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define a simple neural network
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Initialize model
    num_features = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    model = SimpleClassifier(num_features, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = correct / total
    print(f'PyTorch Model Test Accuracy: {accuracy:.4f}')
    
    return model, accuracy

if __name__ == "__main__":
    # Run simple classification example
    clf, sklearn_accuracy, train_coordinates, test_coordinates = simple_classification_example()
    
    # Run PyTorch example
    model, pytorch_accuracy = pytorch_classification_example()
    
    print(f"\nFinal Results:")
    print(f"Random Forest Accuracy: {sklearn_accuracy:.4f}")
    print(f"PyTorch Model Accuracy: {pytorch_accuracy:.4f}") 