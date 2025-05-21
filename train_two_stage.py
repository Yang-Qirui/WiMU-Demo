import torch
import numpy as np
from torch_geometric.nn import GAE
from torch_geometric.data.data import Data
from gnn import *
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import json
from utils import LDPL, plot_adjacency_matrices, plot_fine_tune_losses, plot_pretrain_losses, prune_adjacency_topk_min
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
from torch.nn import TripletMarginLoss
from collections import defaultdict

def pre_train(args):
    
    # Initialize lists to store losses for plotting
    train_recon_losses = []
    train_dist_losses = []
    train_contrast_losses = []
    val_recon_losses = []
    val_dist_losses = []
    val_contrast_losses = []
    
    # Create output directories if they don't exist
    os.makedirs("output/pre_train_plots", exist_ok=True)
    os.makedirs("output/pre_train_adjacency", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    torch.save(A, "./output/adjacency/original_adj.pt")
    # Create graph dataset
    edge_index = A.nonzero().to(device)
    edge_attr = A[edge_index[:, 0], edge_index[:, 1]].to(device)
    x = torch.randn((A.shape[0], args.fp_dim)).to(device)
    graph_dataset = Data(x, edge_index.T, edge_attr)
    
    # Save graph dataset for both pre-training and fine-tuning
    torch.save(graph_dataset, "./output/graph_dataset.pt")
    
    # Load pre-training data
    wifi_inputs = np.load("./data/pre_training/wifi_inputs.npy")
    distance_labels = np.load("./data/pre_training/distance_labels.npy")
    id_mask = np.load("./data/pre_training/id_mask.npy")
    
    # Split into train and validation sets
    dataset_size = len(wifi_inputs)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(0.8 * dataset_size)  # 80% for training, 20% for validation
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
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
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1, total_iters=500)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pre_train_epoch - 500)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[500])
    
    # Loss functions
    loss_fn = torch.nn.MSELoss().to(device)
    recon_loss_fn = torch.nn.BCELoss().to(device)
    contrast_loss_fn = TripletMarginLoss(margin=1.0, p=2).to(device)
    corr_loss_fn = MutualInformationLoss(neg_coor=True).to(device)
    
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
        train_contrast_loss = 0
        
        # Initialize dictionaries to store current epoch predictions
        current_predictions = defaultdict(list)
        
        processed_inputs = set()  # Track processed inputs
        for batch_inputs, batch_labels, batch_id_mask in tqdm(train_loader, desc=f"Pre-training Epoch {epoch}"):
            inputs1, inputs2, inputs3 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
            optimizer.zero_grad()
            
            # Get embeddings and predictions
            prediction_1, recon_A = model(graph_dataset, inputs1)
            prediction_2, _ = model(graph_dataset, inputs2)
            
            # # Get embeddings for contrastive learning
            # emb1 = model.gen_emb(graph_dataset, inputs1)
            # emb2 = model.gen_emb(graph_dataset, inputs2)
            # emb1_norm = torch.nn.functional.normalize(emb1.squeeze(-1), p=2, dim=1)
            # emb2_norm = torch.nn.functional.normalize(emb2.squeeze(-1), p=2, dim=1)
            # similarity = torch.sum(emb1_norm * emb2_norm, dim=1)
            # sim_mean = torch.mean(similarity)
            # dist_mean = torch.mean(batch_labels)
            # sim_centered = similarity - sim_mean
            # dist_centered = batch_labels - dist_mean

            # predict_dist = torch.sqrt(torch.sum((prediction_1 - prediction_2) ** 2, dim=1))
            predict_dist_vector = prediction_2 - prediction_1
            dist_loss = loss_fn(predict_dist_vector, batch_labels)
            recon_loss = recon_loss_fn(recon_A, A)
            
            # contrast_loss = mutual_info_loss(sim_centered, dist_centered)
            loss = 1 * dist_loss + recon_loss # + contrast_loss
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_dist_loss += dist_loss.item()
            # train_contrast_loss += contrast_loss.item()
            
            # Store predictions and ground truth by ID
            for i, id_val in enumerate(batch_id_mask):
                input_key = tuple(inputs1[i].detach().cpu().numpy())  # Convert to tuple for hashing
                if input_key not in processed_inputs:
                    current_predictions[id_val.item()].append(prediction_1[i].detach().cpu().numpy())
                    processed_inputs.add(input_key)
        
        train_epoch_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_dist_loss /= len(train_loader)
        train_contrast_loss /= len(train_loader)
        train_recon_losses.append(train_recon_loss)
        train_dist_losses.append(train_dist_loss)
        train_contrast_losses.append(train_contrast_loss)
        
        # Validation phase
        model.eval()
        val_epoch_loss = 0
        val_recon_loss = 0
        val_dist_loss = 0
        val_contrast_loss = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels, batch_id_mask in val_loader:
                inputs1, inputs2, inputs3 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
                prediction_1, recon_A = model(graph_dataset, inputs1)
                prediction_2, _ = model(graph_dataset, inputs2)
                
                # emb1 = model.gen_emb(graph_dataset, inputs1)
                # emb2 = model.gen_emb(graph_dataset, inputs2)
                # emb3 = model.gen_emb(graph_dataset, inputs3)
                # Get embeddings for contrastive learning
                
                # Calculate similarity between embeddings using cosine similarity
                # emb1_norm = torch.nn.functional.normalize(emb1.squeeze(-1), p=2, dim=1)
                # emb2_norm = torch.nn.functional.normalize(emb2.squeeze(-1), p=2, dim=1)
                # similarity = torch.sum(emb1_norm * emb2_norm, dim=1)
                # sim_mean = torch.mean(similarity)
                # dist_mean = torch.mean(batch_labels)
                # sim_centered = similarity - sim_mean
                # dist_centered = batch_labels - dist_mean
                # sim_centered_divide = sim_centered / torch.std(sim_centered)
                # dist_centered_divide = dist_centered / torch.std(dist_centered)
                # corr = torch.mean(sim_centered_divide * dist_centered_divide)

                # predict_dist = torch.sqrt(torch.sum((prediction_1 - prediction_2) ** 2, dim=1))
                predict_dist_vector = prediction_2 - prediction_1
                dist_loss = loss_fn(predict_dist_vector, batch_labels)
                recon_loss = recon_loss_fn(recon_A, A)
                
                # contrast_loss = mutual_info_loss(sim_centered, dist_centered)
                loss = 1 * dist_loss + recon_loss # + contrast_loss
                val_epoch_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_dist_loss += dist_loss.item()
                # val_contrast_loss += contrast_loss.item()
        
        val_epoch_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_dist_loss /= len(val_loader)
        val_contrast_loss /= len(val_loader)
        val_recon_losses.append(val_recon_loss)
        val_dist_losses.append(val_dist_loss)
        val_contrast_losses.append(val_contrast_loss)
        
        # if val_epoch_loss < best_val_loss:
        #     best_val_loss = val_epoch_loss
        #     best_predictions = current_predictions
            
        #     # Plot predictions
        #     for id_val in best_predictions.keys():
        #         preds = np.array(best_predictions[id_val])
        #         plt.figure(figsize=(15, 10))
        #         plt.scatter(preds[:, 0], preds[:, 1], label=f'ID {id_val} - Predictions', alpha=0.6)
        #         plt.title(f'ID {id_val}')
        #         plt.savefig(f'output/pre_train_plots/best_predictions_epoch_{id_val}.png')
        #         plt.close()
        # # Early stopping check
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            plot_adjacency_matrices(A, recon_A)
            # Save best model
            torch.save(model.state_dict(), "output/pre_trained_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Pre-train Epoch: {epoch}")
            print(f"Train - Total Loss: {train_epoch_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, Dist Loss: {train_dist_loss:.4f}, Contrast Loss: {train_contrast_loss:.4f}")
            print(f"Val - Total Loss: {val_epoch_loss:.4f}, Recon Loss: {val_recon_loss:.4f}, Dist Loss: {val_dist_loss:.4f}, Contrast Loss: {val_contrast_loss:.4f}")
        scheduler.step()
    # Plot training curves
    plot_pretrain_losses(train_recon_losses, train_dist_losses, val_recon_losses, val_dist_losses, train_contrast_losses, val_contrast_losses)
    print("Pre-training completed. Best model saved to output/pre_trained_model.pt")

def load_json_data(data_path, device, A_shape):
    """
    Load and process data from JSON files, handling different frequency bands separately.
    
    Args:
        data_path (str): Path to the directory containing JSON files
        device (torch.device): Device to load tensors to
        A_shape (int): Shape of the adjacency matrix
        
    Returns:
        tuple: (rp_weights, coordinates, timestamps)
    """
    rp_weights = []
    coordinates = []
    timestamps = []
    
    for rp_file in os.listdir(data_path):
        coor = rp_file[:-5]
        coor = ast.literal_eval(coor)
        
        with open(os.path.join(data_path, rp_file), 'r') as f:
            bssid_rssi_dict = json.load(f)
            for timestamp, wifi_records in bssid_rssi_dict.items():
                # Separate 2.4G and 5G signals
                weights = torch.zeros((A_shape,))
                
                for bssid, (rssi, band) in wifi_records.items():
                    if band == '2.4G':
                        weights[int(bssid)] = 1 / (1 + LDPL(rssi, band='2.4G'))
                    elif band == '5G':
                        weights[int(bssid)] = 1 / (1 + LDPL(rssi, band='5G'))
                
                weights /= weights.sum()
                rp_weights.append(weights)
                coordinates.append(coor)
                timestamps.append(int(timestamp))
    
    # Convert to tensors
    rp_weights = torch.stack(rp_weights).float().to(device)
    coordinates = torch.tensor(coordinates).float().to(device)
    timestamps = torch.tensor(timestamps).long().to(device)
    
    return rp_weights, coordinates, timestamps

def fine_tune(args):
    """
    Second stage: Fine-tune the model using labeled data
    """
    # Initialize lists to store losses for plotting
    train_losses = []
    test_errors = []
    
    # Create output directories if they don't exist
    os.makedirs("output/fine_tune_plots", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the same graph dataset used in pre-training
    graph_dataset = torch.load("./output/graph_dataset.pt").to(device)
    A = torch.load("./output/adjacency/original_adj.pt").to(device)
    
    # Initialize models
    gnn = GAE(GCNEncoder(args.fp_dim, args.fp_dim), MLPDecoder(args.fp_dim)).to(device)
    mlp = MyMLP(args.fp_dim, 2).float().to(device)  # Output is coordinates
    model = JointModel(gnn, mlp).to(device)
    
    # Load pre-trained weights
    if os.path.exists("output/pre_trained_model.pt"):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load("output/pre_trained_model.pt"))
    else:
        print("Warning: Pre-trained model not found. Starting from scratch.")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1, total_iters=500)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fine_tune_epoch - 500)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[500])
    
    # Loss functions
    loss_fn = torch.nn.MSELoss().to(device)
    recon_loss_fn = torch.nn.BCELoss().to(device)
    contrast_fn = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    
    # Load training and test data
    train_path = "./data/train_json"
    test_path = "./data/test_json"
    
    print("Loading training data...")
    train_rp_weights, train_coors_tensor, _ = load_json_data(train_path, device, A.shape[0])
    
    print("Loading test data...")
    test_rp_weights, test_coors_tensor, test_timestamps = load_json_data(test_path, device, A.shape[0])
    
    # Calculate normalization parameters
    rp_pos_max = train_coors_tensor.max(dim=0)[0].to(device)
    rp_pos_min = train_coors_tensor.min(dim=0)[0].to(device)
    rp_pos_range = rp_pos_max - rp_pos_min
    rp_pos_range[rp_pos_range == 0] = 1
    rp_pos_range = rp_pos_range.to(device)
    
    # Save normalization parameters
    torch.save({
        'pos_range': rp_pos_range,
        'pos_min': rp_pos_min
    }, './output/norm_params.pt')
    
    # Normalize coordinates
    train_coors_tensor = (train_coors_tensor - rp_pos_min) / rp_pos_range
    test_coors_tensor = (test_coors_tensor - rp_pos_min) / rp_pos_range
    print(f"Position range: {rp_pos_range}, Position min: {rp_pos_min}")
    
    # Create datasets and dataloaders
    train_contrast_dataset = ContrastiveDataset(train_rp_weights.to(device), train_coors_tensor.to(device))
    test_dataset = TensorDataset(test_rp_weights.to(device), test_coors_tensor.to(device), test_timestamps.to(device))
    train_loader = DataLoader(train_contrast_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Fine-tuning loop
    print("Starting fine-tuning...")
    min_test_err = np.inf
    test_std = np.inf
    alpha = 10
    
    for e in range(args.fine_tune_epoch):
        model.train()
        train_epoch_loss = 0
        
        for _train_rp_coors, _train_rp_weights, _train_pos_weights, _train_neg_weights in tqdm(train_loader, desc=f"Fine-tuning Epoch {e}"):
            optimizer.zero_grad()
            _train_rp_weights = _train_rp_weights.to(device)
            _train_pos_weights = _train_pos_weights.to(device)
            _train_neg_weights = _train_neg_weights.to(device)
            _train_rp_coors = _train_rp_coors.to(device)
            # Forward pass
            predict_coors, recon_A = model(graph_dataset, _train_rp_weights)
            
            # Compute losses
            anchor = model.gen_emb(graph_dataset, _train_rp_weights)
            positive = model.gen_emb(graph_dataset, _train_pos_weights)
            negative = model.gen_emb(graph_dataset, _train_neg_weights)
            
            triplet_loss = contrast_fn(anchor, positive, negative)
            recon_loss = recon_loss_fn(recon_A, A)
            loc_loss = loss_fn(predict_coors, _train_rp_coors)
            
            # Combined loss
            loss = recon_loss + alpha * loc_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        scheduler.step()
        train_epoch_loss /= len(train_loader)
        train_losses.append(train_epoch_loss)
        
        # Evaluation
        model.eval()
        test_result = {}
        x_err, y_err, err_std = [], [], []
        test_epoch_error = 0
        
        with torch.no_grad():
            for _test_rp_weights, _test_rp_coors, _test_rp_timestamp in test_loader:
                predict_coors, recon_A = model(graph_dataset, _test_rp_weights)
                delta_coors = (_test_rp_coors - predict_coors) * rp_pos_range
                
                for i in range(len(delta_coors)):
                    _err = delta_coors[i].pow(2).sum().sqrt()
                    _coor = _test_rp_coors[i] * rp_pos_range + rp_pos_min
                    test_result[f"{str(_coor.tolist())}-{_test_rp_timestamp[i].item()}"] = _err.item()
                    x_err.append(delta_coors[i][0].item())
                    y_err.append(delta_coors[i][1].item())
                    test_epoch_error += _err.item()
                    err_std.append(_err.item())
        
        test_epoch_error /= len(test_loader.dataset)
        test_errors.append(test_epoch_error)
        
        # Save best model
        if test_epoch_error <= min_test_err:
            test_result["x_err_std"] = np.std(x_err)
            test_result["y_err_std"] = np.std(y_err)
            
            with open("./output/fine_tuned_test_result.json", 'w') as f:
                json.dump(test_result, f)
            
            min_test_err = test_epoch_error
            test_std = np.std(err_std)
            torch.save(model.state_dict(), "output/fine_tuned_model.pt")
        
        if e % 10 == 0:
            print(f"Fine-tune Epoch: {e}, Train Loss: {train_epoch_loss:.4f}, Test Error: {test_epoch_error:.4f}")
    
    print(f"Fine-tuning completed. Best test error (MAE): {min_test_err:.4f}, Test Error Std: {test_std:.4f}")
    plot_fine_tune_losses(train_losses, test_errors)

def grid_search(param_grid):
        best_error = float('inf')
        best_params = {}
        
        # Generate all parameter combinations
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
        print(f"Starting grid search with {len(param_combinations)} combinations...")
        
        for params in param_combinations:
            print("\nTrying parameters:", params)
            
            # Create args namespace with current parameter set
            args = argparse.Namespace(
                fp_dim=params['fp_dim'],
                lr=params['lr'], 
                batch_size=params['batch_size'],
                patience=params['patience'],
                pre_train_epoch=2000,
                fine_tune_epoch=2000
            )
            
            pre_train(args)
            # Run training with current parameters
            fine_tune(args)
            
            # Load test results to get error
            with open("./output/fine_tuned_test_result.json", 'r') as f:
                results = json.load(f)
            
            # Calculate mean error excluding std values
            errors = [v for k,v in results.items() if k not in ['x_err_std', 'y_err_std']]
            mean_error = sum(errors) / len(errors)
            
            # Update best parameters if better error found
            if mean_error < best_error:
                best_error = mean_error
                best_params = params.copy()
                print(f"New best parameters found! Error: {best_error:.4f}")
                json.dump(best_params, open("./params/best_params.json", 'w'))
        
        print("\nGrid search completed!")
        print("Best parameters:", best_params)
        print(f"Best error: {best_error:.4f}")

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_printoptions(sci_mode=False)
    # Grid search parameters
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--fp_dim", help="dimension of node representations", default=128, type=int)
    arg_parser.add_argument("--lr", help="training learning rate", default=1e-4, type=float)
    arg_parser.add_argument("--pre_train_epoch", help="pre-training epoch number", default=1000, type=int)
    arg_parser.add_argument("--fine_tune_epoch", help="fine-tuning epoch number", default=2000, type=int)
    arg_parser.add_argument("--batch_size", help="batch size for training", default=64, type=int)
    arg_parser.add_argument("--patience", help="patience for early stopping", default=20, type=int)
    arg_parser.add_argument("--pre_train", help="whether to pre-train", action="store_true")
    arg_parser.add_argument("--grid_search", help="whether to grid search", action="store_true")

    args = arg_parser.parse_args()
    
    # Run pre-training
    if args.grid_search:
        param_grid = {
            'fp_dim': [64, 128, 256, 512],
            'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            'batch_size': [4, 8, 16, 32],
            'patience': [20, 30, 40, 50]
        }
        best_params = grid_search(param_grid)
        json.dump(best_params, open("./params/best_params.json", 'w'))
    else:
        if args.pre_train:
            pre_train(args)
        fine_tune(args) 

    