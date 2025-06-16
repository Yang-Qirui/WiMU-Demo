import torch
import numpy as np
from torch_geometric.nn import GAE
from torch_geometric.data.data import Data
from gnn import *
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import json
from utils import LDPL, plot_adjacency_matrices, prune_adjacency_topk_min
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict

def load_classification_data(data_dir, device):
    """Load classification dataset"""
    print(f"Loading classification data from {data_dir}...")
    
    # Load features, labels, and coordinates
    train_features = torch.load(os.path.join(data_dir, 'train_classification', 'train_features.pt')).to(device)
    train_labels = torch.load(os.path.join(data_dir, 'train_classification', 'train_labels.pt')).to(device)
    train_coords = np.load(os.path.join(data_dir, 'train_classification', 'coordinates.npy'))
    
    test_features = torch.load(os.path.join(data_dir, 'test_classification', 'test_features.pt')).to(device)
    test_labels = torch.load(os.path.join(data_dir, 'test_classification', 'test_labels.pt')).to(device)
    test_coords = np.load(os.path.join(data_dir, 'test_classification', 'coordinates.npy'))
    
    # Load dataset statistics
    with open(os.path.join(data_dir, 'train_classification', 'dataset_stats.json'), 'r') as f:
        stats = json.load(f)
    
    # Debug: Print original label information
    print(f"Original train labels - min: {train_labels.min()}, max: {train_labels.max()}")
    print(f"Original test labels - min: {test_labels.min()}, max: {test_labels.max()}")
    print(f"Original unique train labels: {torch.unique(train_labels)}")
    print(f"Original unique test labels: {torch.unique(test_labels)}")
    
    # Get all unique labels from both train and test sets
    all_unique_labels = torch.unique(torch.cat([train_labels, test_labels]))
    print(f"All unique labels: {all_unique_labels}")
    
    # Create label mapping to ensure contiguous labels [0, num_classes-1]
    label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(all_unique_labels)}
    reverse_mapping = {new_label: old_label for old_label, new_label in label_mapping.items()}
    
    print(f"Label mapping: {label_mapping}")
    print(f"Reverse mapping: {reverse_mapping}")
    
    # Apply label mapping
    train_labels_mapped = torch.zeros_like(train_labels)
    for i, label in enumerate(train_labels):
        train_labels_mapped[i] = label_mapping[label.item()]
    
    test_labels_mapped = torch.zeros_like(test_labels)
    for i, label in enumerate(test_labels):
        test_labels_mapped[i] = label_mapping[label.item()]
    
    # Verify mapping
    print(f"Mapped train labels - min: {train_labels_mapped.min()}, max: {train_labels_mapped.max()}")
    print(f"Mapped test labels - min: {test_labels_mapped.min()}, max: {test_labels_mapped.max()}")
    print(f"Mapped unique train labels: {torch.unique(train_labels_mapped)}")
    print(f"Mapped unique test labels: {torch.unique(test_labels_mapped)}")
    
    num_classes = len(all_unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Train set: {train_features.shape[0]} samples, {train_features.shape[1]} features")
    print(f"Test set: {test_features.shape[0]} samples, {test_features.shape[1]} features")
    print(f"Class distribution in train set: {torch.bincount(train_labels_mapped.long())}")
    
    # Save label mapping for future reference
    mapping_info = {
        'label_mapping': label_mapping,
        'reverse_mapping': reverse_mapping,
        'num_classes': num_classes
    }
    
    return train_features, train_labels_mapped, train_coords, test_features, test_labels_mapped, test_coords, stats, mapping_info

def create_graph_dataset(args, device):
    """Create graph dataset from adjacency matrix"""
    print("Creating graph dataset...")
    
    # Load and process adjacency matrix
    A = np.load("./output/data_process/ap_mean.npy")
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    mask = A == 0
    min_val, max_val = np.min(A[~mask]), np.max(A[~mask])
    A[~mask] = (A[~mask] - min_val) / (max_val - min_val)
    mask = A == 0
    k = 10
    A[~mask] = 1 + np.tanh(-k * A[~mask])
    A = prune_adjacency_topk_min(A, k=20)
    A = torch.from_numpy(A).to(device).float()
    
    # Create graph dataset
    edge_index = A.nonzero().to(device)
    edge_attr = A[edge_index[:, 0], edge_index[:, 1]].to(device)
    x = torch.randn((A.shape[0], args.fp_dim)).to(device)
    graph_dataset = Data(x, edge_index.T, edge_attr)
    
    # Save graph dataset
    os.makedirs(f"./{args.save_dir}", exist_ok=True)
    torch.save(graph_dataset, f"./{args.save_dir}/graph_dataset_classification.pt")
    torch.save(A, f"./{args.save_dir}/adjacency_classification.pt")
    
    return graph_dataset, A

def pre_train_gnn(args, device):
    """Pre-train the GNN using distance prediction task"""
    print("Starting GNN pre-training...")
    
    # Create output directories
    os.makedirs(f"{args.save_dir}/pre_train_plots", exist_ok=True)
    os.makedirs(f"{args.save_dir}/pre_train_models", exist_ok=True)
    
    # Load and process adjacency matrix
    A = np.load("./output/data_process/ap_mean.npy")
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    mask = A == 0
    min_val, max_val = np.min(A[~mask]), np.max(A[~mask])
    A[~mask] = (A[~mask] - min_val) / (max_val - min_val)
    mask = A == 0
    k = 10
    A[~mask] = 1 + np.tanh(-k * A[~mask])
    A = prune_adjacency_topk_min(A, k=20)
    A = torch.from_numpy(A).to(device).float()
    
    # Create graph dataset
    edge_index = A.nonzero().to(device)
    edge_attr = A[edge_index[:, 0], edge_index[:, 1]].to(device)
    x = torch.randn((A.shape[0], args.fp_dim)).to(device)
    graph_dataset = Data(x, edge_index.T, edge_attr)
    
    # Save graph dataset
    torch.save(graph_dataset, f"./{args.save_dir}/graph_dataset.pt")
    torch.save(A, f"./{args.save_dir}/adjacency.pt")
    
    # Load pre-training data
    wifi_inputs = np.load("./data/pre_training/wifi_inputs.npy")
    distance_labels = np.load("./data/pre_training/distance_labels.npy")
    id_mask = np.load("./data/pre_training/id_mask.npy")
    
    # Split into train and validation sets
    dataset_size = len(wifi_inputs)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(0.8 * dataset_size)
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_inputs = torch.from_numpy(wifi_inputs[train_indices]).float().to(device)
    train_labels = torch.from_numpy(distance_labels[train_indices]).float().to(device)
    train_id_mask = torch.from_numpy(id_mask[train_indices]).long().to(device)
    val_inputs = torch.from_numpy(wifi_inputs[val_indices]).float().to(device)
    val_labels = torch.from_numpy(distance_labels[val_indices]).float().to(device)
    val_id_mask = torch.from_numpy(id_mask[val_indices]).long().to(device)
    
    train_dataset = TensorDataset(train_inputs, train_labels, train_id_mask)
    val_dataset = TensorDataset(val_inputs, val_labels, val_id_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize models for pre-training
    gnn = GAE(GCNEncoder(args.fp_dim, args.fp_dim), MLPDecoder(args.fp_dim)).to(device)
    mlp = MyMLP(args.fp_dim, 2).float().to(device)  # Output is 2D coordinates
    model = JointModel(gnn, mlp).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pre_train_lr)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1, total_iters=500)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pre_train_epochs - 500)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[500])
    
    # Loss functions
    loss_fn = torch.nn.MSELoss().to(device)
    recon_loss_fn = torch.nn.BCELoss().to(device)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Pre-training for {args.pre_train_epochs} epochs...")
    
    for epoch in range(args.pre_train_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0
        
        for batch_inputs, batch_labels, batch_id_mask in tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}"):
            inputs1, inputs2, inputs3 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
            optimizer.zero_grad()
            
            # Forward pass
            prediction_1, recon_A = model(graph_dataset, inputs1)
            prediction_2, _ = model(graph_dataset, inputs2)
            
            # Calculate losses
            predict_dist_vector = prediction_2 - prediction_1
            dist_loss = loss_fn(predict_dist_vector, batch_labels)
            recon_loss = recon_loss_fn(recon_A, A)
            
            total_loss = dist_loss + recon_loss
            total_loss.backward()
            optimizer.step()
            
            train_epoch_loss += total_loss.item()
        
        train_epoch_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels, batch_id_mask in val_loader:
                inputs1, inputs2, inputs3 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
                prediction_1, recon_A = model(graph_dataset, inputs1)
                prediction_2, _ = model(graph_dataset, inputs2)
                
                predict_dist_vector = prediction_2 - prediction_1
                dist_loss = loss_fn(predict_dist_vector, batch_labels)
                recon_loss = recon_loss_fn(recon_A, A)
                
                total_loss = dist_loss + recon_loss
                val_epoch_loss += total_loss.item()
        
        val_epoch_loss /= len(val_loader)
        scheduler.step()
        
        # Track losses
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Pre-train Epoch {epoch+1}/{args.pre_train_epochs}:")
            print(f"  Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.gnn.state_dict(), f"{args.save_dir}/pre_train_models/best_gnn.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot pre-training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pre-training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_dir}/pre_train_plots/pre_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pre-training completed! Best validation loss: {best_val_loss:.4f}")
    return graph_dataset, A

def train_classification(args):
    """Train classification model with pre-trained GNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(f"{args.save_dir}/classification_plots", exist_ok=True)
    os.makedirs(f"{args.save_dir}/classification_models", exist_ok=True)
    
    # Step 1: Pre-train the GNN if requested
    if args.pre_train:
        graph_dataset, A = pre_train_gnn(args, device)
    else:
        # Load existing pre-trained model
        graph_dataset, A = create_graph_dataset(args, device)
    
    # Load classification data
    train_features, train_labels, train_coords, test_features, test_labels, test_coords, stats, mapping_info = load_classification_data(
        args.data_dir, device)
    
    # Get number of classes from mapping info
    num_classes = mapping_info['num_classes']
    print(f"Using {num_classes} classes for model training")
    
    # Save mapping info for future reference
    with open(f"{args.save_dir}/classification_models/label_mapping.json", 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    # Create data loaders
    train_dataset = ClassificationDataset(train_features, train_labels, train_coords)
    test_dataset = ClassificationDataset(test_features, test_labels, test_coords)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize models
    gnn = GAE(GCNEncoder(args.fp_dim, args.fp_dim), MLPDecoder(args.fp_dim)).to(device)
    
    # Load pre-trained GNN weights if available
    pretrained_path = f"{args.save_dir}/pre_train_models/best_gnn.pt"
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained GNN weights from {pretrained_path}")
        gnn.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("No pre-trained weights found, training from scratch")
    
    classification_head = ClassificationHead(args.fp_dim, num_classes, dropout=args.dropout).to(device)
    model = JointClassificationModel(gnn, classification_head).to(device)
    
    # Optimizer and scheduler for classification training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss functions
    classification_loss_fn = torch.nn.CrossEntropyLoss().to(device)
    recon_loss_fn = torch.nn.BCELoss().to(device)
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_acc = 0.0
    patience_counter = 0
    
    print("Starting classification training...")
    print(f"Training for {args.epochs} epochs with learning rate {args.lr}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels, batch_coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            # Forward pass
            class_logits, recon_A = model(graph_dataset, batch_features)
            
            # Calculate losses
            class_loss = classification_loss_fn(class_logits, batch_labels.long())
            recon_loss = recon_loss_fn(recon_A, A)
            
            # Combined loss
            total_loss = args.class_loss_weight * class_loss + args.recon_weight * recon_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            train_epoch_loss += total_loss.item()
            _, predicted = torch.max(class_logits.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels.long()).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        train_epoch_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        test_epoch_loss = 0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels, batch_coords in test_loader:
                class_logits, recon_A = model(graph_dataset, batch_features)
                
                class_loss = classification_loss_fn(class_logits, batch_labels.long())
                recon_loss = recon_loss_fn(recon_A, A)

                total_loss = args.class_loss_weight * class_loss + args.recon_weight * recon_loss
                
                test_epoch_loss += total_loss.item()
                _, predicted = torch.max(class_logits.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels.long()).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.long().cpu().numpy())
        
        test_accuracy = 100 * test_correct / test_total
        test_epoch_loss /= len(test_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Track metrics
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_epoch_loss)
        test_accuracies.append(test_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            torch.save(model.state_dict(), f"{args.save_dir}/classification_models/best_model.pt")
            patience_counter = 0
            
            # Save detailed results
            results = {
                'epoch': epoch + 1,
                'train_loss': train_epoch_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_epoch_loss,
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(all_labels, all_predictions, output_dict=True),
                'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist()
            }
            
            with open(f"{args.save_dir}/classification_models/best_results.json", 'w') as f:
                json.dump(results, f, indent=2)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies, args.save_dir)
    
    # Final evaluation
    final_evaluation(model, test_loader, graph_dataset, num_classes, args.save_dir, device)
    
    print(f"Training completed! Best test accuracy: {best_test_acc:.2f}%")
    return best_test_acc

def plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies, save_dir):
    """Plot training curves"""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(test_accuracies, label='Test Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/classification_plots/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def final_evaluation(model, test_loader, graph_dataset, num_classes, save_dir, device):
    """Perform final evaluation and generate detailed reports"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_features, batch_labels, batch_coords in test_loader:
            class_logits, _ = model(graph_dataset, batch_features)
            probabilities = torch.softmax(class_logits, dim=1)
            _, predicted = torch.max(class_logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.long().cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to native Python types for JSON serialization
    all_predictions = [int(p) for p in all_predictions]
    all_labels = [int(l) for l in all_labels]
    all_probabilities = [[float(p) for p in prob] for prob in all_probabilities]
    
    # Classification report
    report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{save_dir}/classification_plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save final results with proper type conversion
    final_results = {
        'accuracy': float(accuracy_score(all_labels, all_predictions)),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'true_labels': all_labels,
        'probabilities': all_probabilities
    }
    
    with open(f"{save_dir}/classification_models/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_printoptions(sci_mode=False)
    
    # Argument parser
    arg_parser = argparse.ArgumentParser(description="Train WiFi-based classification model")
    arg_parser.add_argument("--fp_dim", help="dimension of node representations", default=128, type=int)
    arg_parser.add_argument("--lr", help="learning rate", default=1e-3, type=float)
    arg_parser.add_argument("--epochs", help="number of training epochs", default=200, type=int)
    arg_parser.add_argument("--batch_size", help="batch size for training", default=32, type=int)
    arg_parser.add_argument("--patience", help="patience for early stopping", default=20, type=int)
    arg_parser.add_argument("--dropout", help="dropout rate", default=0.5, type=float)
    arg_parser.add_argument("--weight_decay", help="weight decay for optimizer", default=1e-4, type=float)
    arg_parser.add_argument("--recon_weight", help="weight for reconstruction loss", default=1, type=float)
    arg_parser.add_argument("--class_loss_weight", help="weight for reconstruction loss", default=0.1, type=float)
    arg_parser.add_argument("--data_dir", help="directory containing classification data", default="data", type=str)
    arg_parser.add_argument("--save_dir", help="save directory", default="output", type=str)
    arg_parser.add_argument("--pre_train", help="pre-train the GNN", action="store_true")
    arg_parser.add_argument("--pre_train_lr", help="learning rate for pre-training", default=1e-3, type=float)
    arg_parser.add_argument("--pre_train_epochs", help="number of pre-training epochs", default=1000, type=int)
    
    args = arg_parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(f"{args.save_dir}/classification_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train classification model
    best_accuracy = train_classification(args)
    
    print(f"\nTraining completed successfully!")
    print(f"Best test accuracy achieved: {best_accuracy:.2f}%") 