import torch
import numpy as np
from torch_geometric.nn import GAE
from torch_geometric.data.data import Data
from gnn import *
from torch.utils.data import TensorDataset, DataLoader, Dataset
import argparse
import os
import json
from utils import LDPL, plot_adjacency_matrices, plot_fine_tune_losses, plot_pretrain_losses, prune_adjacency_topk_min
from tqdm import tqdm
import ast
from config import LDPL_MODE
import wandb

class DeviceAwareDataset(Dataset):
    """
    Dataset that includes device_id information as additional features
    """
    def __init__(self, rp_weights, coordinates, device_ids, timestamps=None):
        self.rp_weights = rp_weights
        self.coordinates = coordinates
        self.device_ids = device_ids
        self.timestamps = timestamps if timestamps is not None else torch.zeros(len(coordinates))
        
        # Create device_id to index mapping
        unique_device_ids = list(set(device_ids))
        self.device_id_to_idx = {device_id: idx for idx, device_id in enumerate(unique_device_ids)}
        self.num_devices = len(unique_device_ids)
        
        # Convert device_ids to tensor indices
        self.device_indices = torch.tensor([self.device_id_to_idx[device_id] for device_id in device_ids])
        
    def __len__(self):
        return len(self.rp_weights)
    
    def __getitem__(self, idx):
        return (self.rp_weights[idx], self.coordinates[idx], self.device_indices[idx], self.timestamps[idx])

class ContrastiveDatasetWithDevice(Dataset):
    """
    Contrastive dataset that includes device_id information
    """
    def __init__(self, rp_weights, coordinates, device_ids):
        self.rp_weights = rp_weights
        self.coordinates = coordinates
        self.device_ids = device_ids
        
        # Create device_id to index mapping
        unique_device_ids = list(set(device_ids))
        self.device_id_to_idx = {device_id: idx for idx, device_id in enumerate(unique_device_ids)}
        self.num_devices = len(unique_device_ids)
        
        # Convert device_ids to tensor indices
        self.device_indices = torch.tensor([self.device_id_to_idx[device_id] for device_id in device_ids])
        
    def __len__(self):
        return len(self.rp_weights)
    
    def __getitem__(self, idx):
        # Get positive and negative samples for contrastive learning
        pos_idx = idx
        neg_idx = (idx + 1) % len(self.rp_weights)  # Simple negative sampling
        
        return (self.coordinates[idx], self.rp_weights[idx], 
                self.rp_weights[pos_idx], self.rp_weights[neg_idx],
                self.device_indices[idx])

def pre_train(args):
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_name}_pre-train",
            config={
                "fp_dim": args.fp_dim,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "pre_train_epoch": args.pre_train_epoch,
                "beta": args.beta,
                "model_type": "pre_train"
            }
        )
    
    # Initialize lists to store losses for plotting
    train_recon_losses = []
    train_dist_losses = []
    train_l1_losses = []

    val_recon_losses = []
    val_dist_losses = []
    val_l1_losses = []
    
    # Create output directories if they don't exist
    os.makedirs(f"./{args.save_dir}/pre_train_plots", exist_ok=True)
    os.makedirs(f"./{args.save_dir}/pre_train_adjacency", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load adjacency matrix from data-specific directory
    ap_mean_path = f"./{args.save_dir}/data_process/ap_mean.npy"
    A = np.load(ap_mean_path)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    mask = A == 0
    
    min_val, max_val = np.min(A[~mask]), np.max(A[~mask])
    A[~mask] = (A[~mask] - min_val) / (max_val - min_val)
    # 计算非零元素的平均值和方差
    mean_val = np.mean(A[~mask])
    std_val = np.std(A[~mask])
    print(f"非零元素平均值: {mean_val:.4f}")
    print(f"非零元素方差: {std_val:.4f}")
    # assert 0
    mask = A == 0
    # k = 10
    # A[~mask] = 1 + np.tanh(-k * A[~mask])
    A[~mask] = 1 - np.power(A[~mask], 0.25)
    A = prune_adjacency_topk_min(A, k=20)
    A = torch.from_numpy(A).to(device).float()
    
    # Save adjacency matrix to data-specific directory
    os.makedirs(f"./{args.save_dir}/adjacency", exist_ok=True)
    torch.save(A, f"./{args.save_dir}/adjacency/original_adj.pt")
    edge_index = A.nonzero().to(device)
    edge_attr = A[edge_index[:, 0], edge_index[:, 1]].to(device)
    x = torch.randn((A.shape[0], args.fp_dim)).to(device)
    graph_dataset = Data(x, edge_index.T, edge_attr)
    
    torch.save(graph_dataset, f"./{args.save_dir}/graph_dataset.pt")
    
    # Load pre-training data from data-specific directory
    pre_training_dir = f"./{args.save_dir}/pre_training"
    wifi_inputs = np.load(f"{pre_training_dir}/wifi_inputs.npy")
    distance_labels = np.load(f"{pre_training_dir}/distance_labels.npy")
    id_mask = np.load(f"{pre_training_dir}/id_mask.npy")
    
    # Split into train and validation sets
    dataset_size = len(wifi_inputs)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(0.8 * dataset_size)  # 80% for training, 20% for validation
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Save validation indices for consistency with fine-tune
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'split_ratio': 0.8
    }, f"./{args.save_dir}/data_split.pt")
    
    # Create train and validation datasets
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
    
    # Initialize models
    gnn = GAE(GCNEncoder(args.fp_dim, args.fp_dim), MLPDecoder(args.fp_dim)).to(device)
    mlp = MyMLP(args.fp_dim, 2).float().to(device)  # Output is distance
    model = JointModel(gnn, mlp).to(device)
    
    # Optimizer and scheduler with weight decay for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)
    # scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1, total_iters=500)
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pre_train_epoch - 500)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[500, args.pre_train_epoch - 500])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pre_train_epoch)


    # Loss functions
    loss_fn = torch.nn.MSELoss().to(device)
    recon_loss_fn = torch.nn.BCELoss().to(device)
    
    # Pre-training loop
    print("Starting pre-training...")
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(args.pre_train_epoch):
        # Training phase
        model.train()
        train_epoch_loss = 0
        train_recon_loss = 0
        train_dist_loss = 0
        
        # Initialize dictionaries to store current epoch predictions
        for batch_inputs, batch_labels, batch_id_mask in tqdm(train_loader, desc=f"Pre-training Epoch {epoch}"):
            inputs1, inputs2 = batch_inputs[:, 0], batch_inputs[:, 1]
            optimizer.zero_grad()
            
            # Get embeddings and predictions
            prediction_1, recon_A = model(graph_dataset, inputs1)
            prediction_2, _ = model(graph_dataset, inputs2)
            
            predict_dist_vector = prediction_2 - prediction_1
            dist_loss = loss_fn(predict_dist_vector, batch_labels)
            recon_loss = recon_loss_fn(recon_A, A)
            
            loss = args.beta * dist_loss + recon_loss
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_dist_loss += dist_loss.item()

        train_epoch_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_dist_loss /= len(train_loader)
        train_recon_losses.append(train_recon_loss)
        train_dist_losses.append(train_dist_loss)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0
        val_recon_loss = 0
        val_dist_loss = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels, batch_id_mask in val_loader:
                inputs1, inputs2 = batch_inputs[:, 0], batch_inputs[:, 1]
                prediction_1, recon_A = model(graph_dataset, inputs1)
                prediction_2, _ = model(graph_dataset, inputs2)

                predict_dist_vector = prediction_2 - prediction_1
                dist_loss = loss_fn(predict_dist_vector, batch_labels)
                recon_loss = recon_loss_fn(recon_A, A)

                loss = args.beta * dist_loss + recon_loss
                val_epoch_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_dist_loss += dist_loss.item()
        
        val_epoch_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_dist_loss /= len(val_loader)
        val_recon_losses.append(val_recon_loss)
        val_dist_losses.append(val_dist_loss)

        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/total_loss": train_epoch_loss,
                "train/recon_loss": train_recon_loss,
                "train/dist_loss": train_dist_loss,
                "val/total_loss": val_epoch_loss,
                "val/recon_loss": val_recon_loss,
                "val/dist_loss": val_dist_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            plot_adjacency_matrices(A, recon_A)
            # Save best model
            torch.save(model.state_dict(), f"./{args.save_dir}/pre_trained_model.pt")
            
            # Log best model to wandb if enabled
            if args.wandb:
                wandb.save(f"./{args.save_dir}/pre_trained_model.pt")
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Pre-train Epoch: {epoch}")
            print(f"Train - Total Loss: {train_epoch_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, Dist Loss: {train_dist_loss:.4f}")
            print(f"Val - Total Loss: {val_epoch_loss:.4f}, Recon Loss: {val_recon_loss:.4f}, Dist Loss: {val_dist_loss:.4f}")
        scheduler.step()
    
    # Plot training curves
    plot_pretrain_losses(train_recon_losses, train_dist_losses, val_recon_losses, val_dist_losses)
    print(f"Pre-training completed. Best model saved to ./{args.save_dir}/pre_trained_model.pt")
    
    # Finish wandb run if enabled
    if args.wandb:
        wandb.finish()


def load_json_data(data_path, device, A_shape):
    """
    Load and process data from JSON files with new format containing device_id.
    
    Args:
        data_path (str): Path to the directory containing JSON files
        device (torch.device): Device to load tensors to
        A_shape (int): Shape of the adjacency matrix
        
    Returns:
        tuple: (rp_weights, coordinates, timestamps, device_ids)
    """
    rp_weights = []
    coordinates = []
    timestamps = []
    device_ids = []
    
    for rp_file in os.listdir(data_path):
        # Parse filename to extract coordinates and device_id
        # Format: "x_y_device_id_index.json"
        filename_parts = rp_file[:-5].split('_')  # Remove .json and split by _
        
        if len(filename_parts) >= 3:
            # Extract coordinates from filename
            x_coord = float(filename_parts[0])
            y_coord = float(filename_parts[1])
            coor = (x_coord, y_coord)
            
            # Extract device_id (could be multiple parts if device_id contains underscores)
            device_id = '_'.join(filename_parts[2:-1]) if len(filename_parts) > 3 else filename_parts[2]
            
            with open(os.path.join(data_path, rp_file), 'r') as f:
                data = json.load(f)
                
                # Check if data has the new format with device_id and records
                if 'device_id' in data and 'records' in data:
                    # New format: {"device_id": "...", "records": {...}}
                    device_id_from_file = data['device_id']
                    records = data['records']
                    
                    # Process each timestamp in records
                    for timestamp, wifi_records in records.items():
                        # Separate 2.4G and 5G signals
                        weights = torch.zeros((A_shape,))
                        
                        for bssid, (rssi, band) in wifi_records.items():    
                            weights[int(bssid)] = 1 / (1 + LDPL(rssi, band=band, mode=LDPL_MODE))
                        
                        weights /= weights.sum()
                        rp_weights.append(weights)
                        coordinates.append(coor)
                        timestamps.append(int(timestamp))
                        device_ids.append(device_id_from_file)
                else:
                    # Fallback to old format for compatibility
                    for timestamp, wifi_records in data.items():
                        # Separate 2.4G and 5G signals
                        weights = torch.zeros((A_shape,))
                        
                        for bssid, (rssi, band) in wifi_records.items():    
                            weights[int(bssid)] = 1 / (1 + LDPL(rssi, band=band, mode=LDPL_MODE))
                        
                        weights /= weights.sum()
                        rp_weights.append(weights)
                        coordinates.append(coor)
                        timestamps.append(int(timestamp))
                        device_ids.append(device_id)
        else:
            # Fallback to old format parsing
            coor = rp_file[:-5]
            coor = ast.literal_eval(coor)
            
            with open(os.path.join(data_path, rp_file), 'r') as f:
                bssid_rssi_dict = json.load(f)
                for timestamp, wifi_records in bssid_rssi_dict.items():
                    # Separate 2.4G and 5G signals
                    weights = torch.zeros((A_shape,))
                    
                    for bssid, (rssi, band) in wifi_records.items():    
                        weights[int(bssid)] = 1 / (1 + LDPL(rssi, band=band, mode=LDPL_MODE))
                    
                    weights /= weights.sum()
                    rp_weights.append(weights)
                    coordinates.append(coor)
                    timestamps.append(int(timestamp))
                    device_ids.append("unknown")  # Default device_id for old format
    
    # Convert to tensors
    rp_weights = torch.stack(rp_weights).float().to(device)
    coordinates = torch.tensor(coordinates).float().to(device)
    timestamps = torch.tensor(timestamps).long().to(device)
    
    return rp_weights, coordinates, timestamps, device_ids

def fine_tune(args):
    """
    Second stage: Fine-tune the model using labeled data
    """
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_name}_fine-tune",
            config={
                "fp_dim": args.fp_dim,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "fine_tune_epoch": args.fine_tune_epoch,
                "beta": args.beta,
                "model_type": "fine_tune"
            }
        )
    
    # Initialize lists to store losses for plotting
    train_losses = []
    val_losses = []
    test_errors = []
    
    # Create output directories if they don't exist
    os.makedirs(f"./{args.save_dir}/fine_tune_plots", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the same graph dataset used in pre-training
    graph_dataset = torch.load(f"./{args.save_dir}/graph_dataset.pt", weights_only=False).to(device)
    A = torch.load(f"./{args.save_dir}/adjacency/original_adj.pt", weights_only=False).to(device)
    
    # Initialize models
    gnn = GAE(GCNEncoder(args.fp_dim, args.fp_dim), MLPDecoder(args.fp_dim)).to(device)
    mlp = MyMLP(args.fp_dim, 2).float().to(device)  # Output is coordinates
    model = JointModel(gnn, mlp).to(device)
    
    # Load pre-trained weights
    if os.path.exists(f"./{args.save_dir}/pre_trained_model.pt"):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(f"./{args.save_dir}/pre_trained_model.pt", weights_only=False))
    else:
        print("Warning: Pre-trained model not found. Starting from scratch.")
    
    # Optimizer and scheduler with weight decay for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_lambda)
    # scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1, total_iters=500)
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fine_tune_epoch - 500)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[500])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fine_tune_epoch)

    # Loss functions
    loss_fn = torch.nn.MSELoss().to(device)
    recon_loss_fn = torch.nn.BCELoss().to(device)
    
    # Load training and test data from data-specific directory
    train_path = f"./{args.save_dir}/train_json"
    test_path = f"./{args.save_dir}/test_json"
    
    print(f"Loading training data from: {train_path}")
    train_rp_weights, train_coors_tensor, _, train_device_ids = load_json_data(train_path, device, A.shape[0])
    
    if args.merge_samples:
        # Merge samples with same coordinates only for training data
        print("Merging training samples with same coordinates...")
        train_rp_weights, train_coors_tensor, train_timestamps_merged, train_device_ids = merge_samples(
            train_rp_weights, train_coors_tensor, torch.zeros(len(train_rp_weights)), train_device_ids
        )
    
    print(f"Loading test data from: {test_path}")
    test_rp_weights, test_coors_tensor, test_timestamps, test_device_ids = load_json_data(test_path, device, A.shape[0])
    
    # Create independent train/validation split for fine-tuning
    print("Creating independent train/validation split for fine-tuning...")
    dataset_size = len(train_rp_weights)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    print(f"Fine-tuning dataset size: {dataset_size}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Save the fine-tuning split for future reference
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'split_ratio': 0.8,
        'dataset_size': dataset_size
    }, f"./{args.save_dir}/fine_tune_data_split.pt")
    
    # Split training data using consistent indices
    train_rp_weights_split = train_rp_weights[train_indices]
    train_coors_tensor_split = train_coors_tensor[train_indices]
    val_rp_weights = train_rp_weights[val_indices]
    val_coors_tensor = train_coors_tensor[val_indices]
    
    # Split device_ids using the same indices
    train_device_ids_split = [train_device_ids[i] for i in train_indices]
    val_device_ids = [train_device_ids[i] for i in val_indices]
    
    print(f"Test samples: {len(test_rp_weights)}")
    
    # Calculate normalization parameters using only training data
    rp_pos_max = train_coors_tensor_split.max(dim=0)[0].to(device)
    rp_pos_min = train_coors_tensor_split.min(dim=0)[0].to(device)
    rp_pos_range = rp_pos_max - rp_pos_min
    rp_pos_range[rp_pos_range == 0] = 1
    rp_pos_range = rp_pos_range.to(device)
    
    # Save normalization parameters
    torch.save({
        'pos_range': rp_pos_range,
        'pos_min': rp_pos_min
    }, f"./{args.save_dir}/norm_params.pt")
    
    # Normalize coordinates
    train_coors_tensor_split = (train_coors_tensor_split - rp_pos_min) / rp_pos_range
    val_coors_tensor = (val_coors_tensor - rp_pos_min) / rp_pos_range
    test_coors_tensor = (test_coors_tensor - rp_pos_min) / rp_pos_range
    print(f"Position range: {rp_pos_range}, Position min: {rp_pos_min}")
    
    # Create datasets and dataloaders with device_id information
    train_contrast_dataset = ContrastiveDatasetWithDevice(train_rp_weights_split.to(device), train_coors_tensor_split.to(device), train_device_ids_split)
    val_dataset = DeviceAwareDataset(val_rp_weights.to(device), val_coors_tensor.to(device), val_device_ids)
    test_dataset = DeviceAwareDataset(test_rp_weights.to(device), test_coors_tensor.to(device), test_device_ids, test_timestamps.to(device))
    train_loader = DataLoader(train_contrast_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Fine-tuning loop
    print("Starting fine-tuning...")
    min_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    min_test_err = np.inf
    test_std = np.inf
    train_std = np.inf
    
    for e in range(args.fine_tune_epoch):
        model.train()
        train_epoch_loss = 0
        train_error = 0
        train_errors = []
        
        for _train_rp_coors, _train_rp_weights, _train_pos_weights, _train_neg_weights, _train_device_indices in tqdm(train_loader, desc=f"Fine-tuning Epoch {e}"):
            optimizer.zero_grad()
            _train_rp_weights = _train_rp_weights.to(device)
            _train_pos_weights = _train_pos_weights.to(device)
            _train_neg_weights = _train_neg_weights.to(device)
            _train_rp_coors = _train_rp_coors.to(device)
            _train_device_indices = _train_device_indices.to(device)
            
            # Forward pass
            predict_coors, recon_A = model(graph_dataset, _train_rp_weights)

            recon_loss = recon_loss_fn(recon_A, A)
            loc_loss = loss_fn(predict_coors, _train_rp_coors)
            
            # Calculate standard deviation of location errors during training
            delta_coors = (_train_rp_coors - predict_coors) * rp_pos_range
            for i in range(len(delta_coors)):
                _err = delta_coors[i].pow(2).sum().sqrt()
                train_real_coor = _train_rp_coors[i] * rp_pos_range + rp_pos_min
                predict_real_coor = predict_coors[i] * rp_pos_range + rp_pos_min
                train_errors.append((f"[{train_real_coor[0]}, {train_real_coor[1]}]-[{predict_real_coor[0]}, {predict_real_coor[1]}]-[{predict_coors[i][0]}, {predict_coors[i][1]}]", _err.item()))
                train_error += _err.item()
            loss = loc_loss
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        scheduler.step()
        train_std = np.std([train_errors[i][1] for i in range(len(train_errors))])
        train_epoch_loss /= len(train_loader)
        train_error /= len(train_loader)
        train_losses.append(train_epoch_loss)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0
        val_errors = []
        val_error = 0
        
        with torch.no_grad():
            for _val_rp_weights, _val_rp_coors, _val_device_indices, _val_timestamps in val_loader:
                _val_rp_weights = _val_rp_weights.to(device)
                _val_rp_coors = _val_rp_coors.to(device)
                _val_device_indices = _val_device_indices.to(device)
                _val_timestamps = _val_timestamps.to(device)
                
                # Forward pass
                predict_coors, recon_A = model(graph_dataset, _val_rp_weights)
                
                recon_loss = recon_loss_fn(recon_A, A)
                loc_loss = loss_fn(predict_coors, _val_rp_coors)
                
                # Calculate validation errors
                delta_coors = (_val_rp_coors - predict_coors) * rp_pos_range
                for i in range(len(delta_coors)):
                    _err = delta_coors[i].pow(2).sum().sqrt()
                    val_real_coor = _val_rp_coors[i] * rp_pos_range + rp_pos_min
                    predict_real_coor = predict_coors[i] * rp_pos_range + rp_pos_min
                    val_errors.append((f"[{val_real_coor[0]}, {val_real_coor[1]}]-[{predict_real_coor[0]}, {predict_real_coor[1]}]", _err.item()))
                    val_error += _err.item()
                loss = loc_loss
                val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_loader)
        val_error /= len(val_loader)
        val_std = np.std([val_errors[i][1] for i in range(len(val_errors))])
        val_losses.append(val_epoch_loss)
        
        # Log training and validation metrics to wandb if enabled
        if args.wandb:
            wandb.log({
                "epoch": e,
                "train/loss": train_epoch_loss,
                "train/std": train_std,
                "train/error": train_error,
                "val/loss": val_epoch_loss,
                "val/std": val_std,
                "val/error": val_error,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Early stopping based on validation loss
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            patience_counter = 0
            # Save best model based on validation loss
            torch.save(model.state_dict(), f"./{args.save_dir}/fine_tuned_model.pt")
            
            # Log best model to wandb if enabled
            if args.wandb:
                wandb.save(f"./{args.save_dir}/fine_tuned_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {e}")
                break
        
        train_result = {}
        for i in range(len(train_errors)):
            train_result[train_errors[i][0]] = train_errors[i][1]
        train_result["train_std"] = train_std
        
        with open(f"./{args.save_dir}/fine_tuned_train_result.json", 'w') as f:
            json.dump(train_result, f)
        
        # Test phase
        model.eval()
        test_result = {}
        x_err, y_err, test_err_std = [], [], []
        test_epoch_error = 0
        
        with torch.no_grad():
            for _test_rp_weights, _test_rp_coors, _test_device_indices, _test_rp_timestamp in test_loader:
                predict_coors, recon_A = model(graph_dataset, _test_rp_weights)
                delta_coors = (_test_rp_coors - predict_coors) * rp_pos_range
                for i in range(len(delta_coors)):
                    _err = delta_coors[i].pow(2).sum().sqrt()
                    _coor = _test_rp_coors[i] * rp_pos_range + rp_pos_min
                    test_result[f"{str(_coor.tolist())}-{_test_rp_timestamp[i].item()}"] = _err.item()
                    x_err.append(delta_coors[i][0].item())
                    y_err.append(delta_coors[i][1].item())
                    test_epoch_error += _err.item()
                    test_err_std.append(_err.item())
            test_std = np.std(test_err_std)
        test_epoch_error /= len(test_loader.dataset)
        test_errors.append(test_epoch_error)
        
        # Log test metrics to wandb if enabled
        if args.wandb:
            wandb.log({
                "test/error": test_epoch_error,
                "test/std": test_std,
                "test/x_err_std": np.std(x_err),
                "test/y_err_std": np.std(y_err)
            })
        
        # Update best test error tracking
        if test_epoch_error <= min_test_err:
            test_result["x_err_std"] = np.std(x_err)
            test_result["y_err_std"] = np.std(y_err)
            with open(f"./{args.save_dir}/fine_tuned_test_result.json", 'w') as f:
                json.dump(test_result, f)
            min_test_err = test_epoch_error
        
        if e % 10 == 0:
            print(f"Fine-tune Epoch: {e}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Test Error: {test_epoch_error:.4f}, Train Std: {train_std:.4f}, Val Std: {val_std:.4f}, Test Std: {test_std:.4f}")
    
    print(f"Fine-tuning completed. Best validation loss: {min_val_loss:.4f}, Best test error (MAE): {min_test_err:.4f}, Test Error Std: {test_std:.4f}, Train Std: {train_std:.4f}")
    plot_fine_tune_losses(train_losses, val_losses, test_errors)
    best_model = JointModel(gnn, mlp).to(device)
    best_model.load_state_dict(torch.load(f"./{args.save_dir}/fine_tuned_model.pt", weights_only=False))
    compute_and_save_embeddings(best_model, graph_dataset, train_loader, f"./{args.save_dir}/train_embeddings.pt", "train", device)
    compute_and_save_embeddings(best_model, graph_dataset, test_loader, f"./{args.save_dir}/test_embeddings.pt", "test", device)
    
    # Finish wandb run if enabled
    if args.wandb:
        wandb.finish()

def merge_samples(rp_weights, coordinates, timestamps, device_ids, tolerance=1e-6):
    """
    Merge samples that have the same or very close coordinates.
    
    Args:
        rp_weights (torch.Tensor): WiFi signal weights tensor
        coordinates (torch.Tensor): Coordinate tensor  
        timestamps (torch.Tensor): Timestamp tensor
        device_ids (list): List of device IDs
        tolerance (float): Tolerance for coordinate comparison
        
    Returns:
        tuple: (merged_rp_weights, merged_coordinates, merged_timestamps, merged_device_ids)
    """
    print("Merging samples with same coordinates...")
    
    # Create a dictionary to group samples by coordinates
    coord_groups = {}
    
    for i in range(len(coordinates)):
        coord_key = tuple(coordinates[i].cpu().numpy().round(6))  # Round to avoid floating point issues
        
        if coord_key not in coord_groups:
            coord_groups[coord_key] = {
                'indices': [],
                'weights': [],
                'timestamps': [],
                'device_ids': []
            }
        
        coord_groups[coord_key]['indices'].append(i)
        coord_groups[coord_key]['weights'].append(rp_weights[i])
        coord_groups[coord_key]['timestamps'].append(timestamps[i])
        coord_groups[coord_key]['device_ids'].append(device_ids[i])
    
    # Merge samples for each coordinate group
    merged_rp_weights = []
    merged_coordinates = []
    merged_timestamps = []
    merged_device_ids = []
    
    for coord_key, group in coord_groups.items():
        if len(group['indices']) == 1:
            # Single sample, no merging needed
            merged_rp_weights.append(group['weights'][0])
            merged_coordinates.append(torch.tensor(coord_key))
            merged_timestamps.append(group['timestamps'][0])
            merged_device_ids.append(group['device_ids'][0])
        else:
            # Multiple samples, merge them
            # Take average of WiFi weights
            avg_weights = torch.stack(group['weights']).mean(dim=0)
            merged_rp_weights.append(avg_weights)
            
            # Use the coordinate key as the merged coordinate
            merged_coordinates.append(torch.tensor(coord_key))
            
            # Use the median timestamp
            median_timestamp = torch.stack(group['timestamps']).median()
            merged_timestamps.append(median_timestamp)
            
            # Use the most common device_id, or first one if tied
            device_count = {}
            for device_id in group['device_ids']:
                device_count[device_id] = device_count.get(device_id, 0) + 1
            most_common_device = max(device_count.items(), key=lambda x: x[1])[0]
            merged_device_ids.append(most_common_device)
            
            print(f"Merged {len(group['indices'])} samples at coordinate {coord_key}")
    
    # Convert back to tensors and ensure they're on the same device as input
    device = rp_weights.device
    merged_rp_weights = torch.stack(merged_rp_weights).to(device)
    merged_coordinates = torch.stack(merged_coordinates).to(device)
    merged_timestamps = torch.stack(merged_timestamps).to(device)
    
    print(f"Original samples: {len(rp_weights)}, Merged samples: {len(merged_rp_weights)}")
    
    return merged_rp_weights, merged_coordinates, merged_timestamps, merged_device_ids

def grid_search(param_grid, args):
        # Initialize wandb for grid search if enabled
        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                name="grid_search",
                config={
                    "search_type": "grid_search",
                    "param_grid": param_grid
                }
            )
        
        best_error = float('inf')
        best_params = {}
        
        # Generate all parameter combinations
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
        print(f"Starting grid search with {len(param_combinations)} combinations...")
        
        for i, params in enumerate(param_combinations):
            try:
                print(f"\nTrying parameters {i+1}/{len(param_combinations)}:", params)
                
                # Create args namespace with current parameter set
                dir_path = f"output/{params['fp_dim']}-{params['lr']}-{params['batch_size']}-{params['patience']}"
                os.makedirs(dir_path, exist_ok=True)
                args = argparse.Namespace(
                    fp_dim=params['fp_dim'],
                    lr=params['lr'], 
                    batch_size=params['batch_size'],
                    patience=params['patience'],
                    pre_train_epoch=2000,
                    fine_tune_epoch=2000,
                    save_dir = dir_path,
                    beta = 10
                )
                
                pre_train(args)
                # Run training with current parameters
                fine_tune(args)
                
                # Load test results to get error
                with open(f"./output/fine_tuned_test_result.json", 'r') as f:
                    results = json.load(f)
                
                # Calculate mean error excluding std values
                errors = [v for k,v in results.items() if k not in ['x_err_std', 'y_err_std']]
                mean_error = sum(errors) / len(errors)
                
                # Log grid search results to wandb if enabled
                if args.wandb:
                    wandb.log({
                        "grid_search/combination": i + 1,
                        "grid_search/mean_error": mean_error,
                        "grid_search/fp_dim": params['fp_dim'],
                        "grid_search/lr": params['lr'],
                        "grid_search/batch_size": params['batch_size'],
                        "grid_search/patience": params['patience']
                    })
                
                # Update best parameters if better error found
                if mean_error < best_error:
                    best_error = mean_error
                    best_params = params.copy()
                    print(f"New best parameters found! Error: {best_error:.4f}")
                    json.dump(best_params, open(f"./output/best_params.json", 'w'))
                    
                    # Log best parameters to wandb if enabled
                    if args.wandb:
                        wandb.log({
                            "grid_search/best_error": best_error,
                            "grid_search/best_fp_dim": best_params['fp_dim'],
                            "grid_search/best_lr": best_params['lr'],
                            "grid_search/best_batch_size": best_params['batch_size'],
                            "grid_search/best_patience": best_params['patience']
                        })
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                if args.wandb:
                    wandb.log({
                        "grid_search/combination": i + 1,
                        "grid_search/error": str(e)
                    })
        
        print("\nGrid search completed!")
        print("Best parameters:", best_params)
        print(f"Best error: {best_error:.4f}")
        
        # Finish wandb run if enabled
        if args.wandb:
            wandb.finish()

def compute_and_save_embeddings(model, graph_dataset, data_loader, save_path, phase_name, device):
    """
    Compute embeddings for all samples in the dataset and save them.
    
    Args:
        model: The trained model
        graph_dataset: Graph data
        data_loader: DataLoader containing the dataset
        save_path: Path to save embeddings
        phase_name: Name of the phase (e.g., 'train', 'test', 'val')
        device: Device to run computation on
    """
    model.eval()
    all_embeddings = []
    all_coordinates = []
    all_timestamps = []
    all_device_ids = []
    
    print(f"Computing embeddings for {phase_name} dataset...")
    existing_coords = set()
    
    # Get device_id mapping from dataset
    dataset = data_loader.dataset
    device_id_mapping = None
    if hasattr(dataset, 'device_id_to_idx'):
        # Create reverse mapping from idx to device_id
        device_id_mapping = {idx: device_id for device_id, idx in dataset.device_id_to_idx.items()}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"Computing {phase_name} embeddings")):
            
            # Fine-tuning data format with device_id
            if len(batch_data) == 4:
                # Format: (weights, coordinates, device_indices, timestamps)
                batch_weights, batch_coords, batch_device_indices, batch_timestamps = batch_data
                
                # For each sample in batch
                for i in range(len(batch_coords)):
                    # Use coordinate tuple as label
                    coord_label = tuple(batch_coords[i].cpu().numpy())
                
                    # Only store first occurrence of each coordinate
                    if coord_label not in existing_coords:
                        all_coordinates.append(batch_coords[i].cpu())
                        all_timestamps.append(batch_timestamps[i].cpu())
                        all_embeddings.append(model.gen_emb(graph_dataset, batch_weights[i].to(device)).cpu())
                        
                        # Get device_id from device_indices
                        device_idx = batch_device_indices[i].item()
                        if device_id_mapping is not None and device_idx in device_id_mapping:
                            all_device_ids.append(device_id_mapping[device_idx])
                        else:
                            all_device_ids.append("unknown")
                        
                        existing_coords.add(coord_label)
            elif len(batch_data) == 5: # training data format (ContrastiveDatasetWithDevice)
                # Format: (coords, weights, pos_weights, neg_weights, device_indices)
                batch_coords, batch_weights, pos_weights, neg_weights, batch_device_indices = batch_data
                
                # For each sample in batch
                for i in range(len(batch_coords)):
                    # Use coordinate tuple as label
                    coord_label = tuple(batch_coords[i].cpu().numpy())
                
                    # Only store first occurrence of each coordinate
                    if coord_label not in existing_coords:
                        all_coordinates.append(batch_coords[i].cpu())
                        # Use current timestamp as placeholder since ContrastiveDataset doesn't have timestamps
                        all_timestamps.append(torch.tensor(0).cpu())
                        all_embeddings.append(model.gen_emb(graph_dataset, batch_weights[i].to(device)).cpu())
                        
                        # Get device_id from device_indices
                        device_idx = batch_device_indices[i].item()
                        if device_id_mapping is not None and device_idx in device_id_mapping:
                            all_device_ids.append(device_id_mapping[device_idx])
                        else:
                            all_device_ids.append("unknown")
                        
                        existing_coords.add(coord_label)
        
        if len(all_embeddings) > 0:
            all_coordinates = torch.stack(all_coordinates, dim=0)
            all_timestamps = torch.stack(all_timestamps, dim=0)
            all_embeddings = torch.stack(all_embeddings, dim=0)
        else:
            print(f"Warning: No embeddings found for {phase_name} dataset")
            return None
    
    # Save embeddings and metadata
    embeddings_data = {
        'embeddings': all_embeddings.numpy(),
        'coordinates': all_coordinates.numpy(),
        'timestamps': all_timestamps.numpy(),
        'device_ids': all_device_ids,
        'phase': phase_name,
        'num_samples': len(all_embeddings),
        'embedding_dim': all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else all_embeddings.shape[0]
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as both .pt and .npz for flexibility
    torch.save(embeddings_data, save_path.replace('.npz', '.pt'))
    np.savez(save_path, **embeddings_data)
    
    print(f"Saved {phase_name} embeddings: {all_embeddings.shape} to {save_path}")
    return embeddings_data

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_printoptions(sci_mode=False)
    # Grid search parameters
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--fp_dim", help="dimension of node representations", default=128, type=int)
    arg_parser.add_argument("--lr", help="training learning rate", default=1e-4, type=float)
    arg_parser.add_argument("--pre_train_epoch", help="pre-training epoch number", default=1000, type=int)
    arg_parser.add_argument("--fine_tune_epoch", help="fine-tuning epoch number", default=750, type=int)
    arg_parser.add_argument("--batch_size", help="batch size for training", default=16, type=int)
    arg_parser.add_argument("--patience", help="patience for early stopping", default=20, type=int)
    arg_parser.add_argument("--pre_train", help="whether to pre-train", action="store_true")
    arg_parser.add_argument("--grid_search", help="whether to grid search", action="store_true")
    arg_parser.add_argument("--save_dir", help="save directory", default="output", type=str)
    arg_parser.add_argument("--l2_lambda", help="L2 regularization strength (weight decay)", default=1e-5, type=float)
    arg_parser.add_argument("--beta", help="beta for relative localization loss", default=10, type=float)
    arg_parser.add_argument("--wandb", help="enable wandb logging", action="store_true")
    arg_parser.add_argument("--wandb_project", help="wandb project name", default="wimu-demo", type=str)
    arg_parser.add_argument("--merge_samples", help="whether to merge samples with same coordinates", action="store_true")
    arg_parser.add_argument("--wandb_name", help="wandb run name", default="default", type=str)

    args = arg_parser.parse_args()
    os.makedirs(f"./{args.save_dir}", exist_ok=True)
    
    # Run pre-training
    if args.grid_search:
        param_grid = {
            'fp_dim': [64, 128, 256, 512],
            'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            'batch_size': [4, 8, 16, 32],
            'patience': [20, 30, 40, 50]
        }
        best_params = grid_search(param_grid, args)
        json.dump(best_params, open(f"./{args.save_dir}/best_params.json", 'w'))
    else:
        if args.pre_train:
            pre_train(args)
        fine_tune(args) 
    