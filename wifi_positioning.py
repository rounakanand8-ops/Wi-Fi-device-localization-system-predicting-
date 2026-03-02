import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os
from collections import OrderedDict

# ────────────────────────────────────────────────
#  CONFIGURATION – CHANGE FILENAMES IF NEEDED
# ────────────────────────────────────────────────
ENVIRONMENTS = OrderedDict([
    ("Corridor", {
        "train": "corridor_train.csv",
        "test":  "corridor_test.csv",
        "target_mae": 1.85,
        "lr": 0.002,
        "patience": 80,
        "epochs": 600
    }),
    ("Theatre", {
        "train": "theatre_train.csv",
        "test":  "theatre_test.csv",
        "target_mae": 1.23,
        "lr": 0.0015,
        "patience": 100,
        "epochs": 700
    }),
    ("Office", {
        "train": "office_train.csv",
        "test":  "office_test.csv",
        "target_mae": 1.98,
        "lr": 0.0018,
        "patience": 90,
        "epochs": 650
    })
])

# ────────────────────────────────────────────────
#  DATA PREPROCESSING + FEATURE ENGINEERING
# ────────────────────────────────────────────────
def prepare_data(path, scaler=None, fit_scaler=False):
    """
    Load CSV, convert units, create LOS features, derive distance & stats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading {os.path.basename(path)} ...")
    t0 = time.time()

    df = pd.read_csv(path)

    # RTT → meters
    rtt_cols = [f'AP{i} RTT(mm)' for i in range(1, 6)]
    for c in rtt_cols:
        df[c] = df[c] / 1000.0

    rss_cols = [f'AP{i} RSS(dBm)' for i in range(1, 6)]

    X_raw = df[rtt_cols + rss_cols].to_numpy(dtype=np.float32)

    # LOS one-hot
    los = np.zeros((len(df), 5), dtype=np.float32)
    for i, v in enumerate(df['LOS APs'].fillna('')):
        s = str(v).strip()
        if s and s.lower() not in ['nan', 'none', '']:
            try:
                aps = [int(x.strip()) for x in s.split(',') if x.strip().isdigit()]
                for ap in aps:
                    if 1 <= ap <= 5:
                        los[i, ap-1] = 1.0
            except:
                pass

    # Derived features
    rtt_dist = X_raw[:, :5] / 2.0

    X = np.hstack([
        X_raw,
        los,
        X_raw[:, :5] * los,
        X_raw[:, 5:] * los,
        rtt_dist,
        np.mean(X_raw[:, :5], axis=1, keepdims=True),
        np.mean(X_raw[:, 5:], axis=1, keepdims=True),
        np.std(X_raw[:, :5], axis=1, keepdims=True),
    ])

    y = df[['X', 'Y']].to_numpy(dtype=np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)

    print(f"Prepared in {time.time()-t0:.1f}s  |  shape: {X.shape}")
    return X, y, scaler


# ────────────────────────────────────────────────
#  MODEL
# ────────────────────────────────────────────────
class PositionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = [1024, 512, 256, 128, 64]
        
        layers = []
        prev = input_dim
        
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.12))
            prev = h
        
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────
#  TRAINING
# ────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs, lr, patience, env_name, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5)
    
    best_val = float('inf')
    counter = 0
    best_path = f"best_model_{env_name}.pth"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:4d} | Train MSE: {train_loss:8.6f} | Val MSE: {val_loss:8.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    
    return model


# ────────────────────────────────────────────────
#  EVALUATION
# ────────────────────────────────────────────────
def evaluate_model(model, loader, env_name, target_mae, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            targets.append(yb.numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    # Print sample of predicted (X, Y) vs actual (first 5 for demonstration)
    print(f"\n{env_name} - Sample Predicted Locations (X, Y):")
    print("Sample # | Predicted (X, Y) | Actual (X, Y)")
    print("─────────┼──────────────────┼──────────────")
    for i in range(min(5, len(preds))):
        print(f"{i+1:<9} | ({preds[i][0]:.3f}, {preds[i][1]:.3f}) | ({targets[i][0]:.3f}, {targets[i][1]:.3f})")
    
    # Save all predictions to CSV
    pred_df = pd.DataFrame({
        'X_pred': preds[:, 0],
        'Y_pred': preds[:, 1],
        'X_true': targets[:, 0],
        'Y_true': targets[:, 1]
    })
    pred_df.to_csv(f"{env_name}_predictions.csv", index=False)
    print(f"\nAll predicted (X, Y) saved to {env_name}_predictions.csv")
    
    errors = np.sqrt(np.sum((preds - targets)**2, axis=1))
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    status = "✓ BEATEN" if mae <= target_mae else f"✗ NOT YET (need ≤ {target_mae})"
    
    print(f"\n{env_name} Results:")
    print(f"  MAE  = {mae:.3f} m   (target: {target_mae} m)  → {status}")
    print(f"  RMSE = {rmse:.3f} m")
    
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    plt.figure(figsize=(9,5))
    plt.plot(sorted_errors, cdf, label=f'{env_name} model')
    plt.axvline(target_mae, color='r', linestyle='--', label=f'Target ({target_mae} m)')
    plt.xlabel('Error (m)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{env_name} - Error CDF')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return mae, rmse, status


# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    PIN_MEMORY = device == 'cuda'
    WORKERS = 0 if device == 'cpu' else 2
    
    summary = {}
    
    for env, cfg in ENVIRONMENTS.items():
        print(f"\n{'═'*65}")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"{'═'*65}\n")
        
        train_file = cfg["train"]
        test_file = cfg["test"]
        
        # Prepare data
        train_X, train_y, scaler = prepare_data(train_file, fit_scaler=True)
        test_X,  test_y,  _      = prepare_data(test_file, scaler=scaler)
        
        # Validation split
        val_frac = 0.10
        val_size = int(len(train_X) * val_frac)
        
        train_Xv = train_X[:-val_size]
        train_yv = train_y[:-val_size]
        val_X    = train_X[-val_size:]
        val_y    = train_y[-val_size:]
        
        train_ds = TensorDataset(torch.from_numpy(train_Xv), torch.from_numpy(train_yv))
        val_ds   = TensorDataset(torch.from_numpy(val_X),    torch.from_numpy(val_y))
        test_ds  = TensorDataset(torch.from_numpy(test_X),   torch.from_numpy(test_y))
        
        BATCH_SIZE = 1024 if device == 'cuda' else 768
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=WORKERS, pin_memory=PIN_MEMORY)
        
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                                  num_workers=WORKERS, pin_memory=PIN_MEMORY)
        
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,
                                  num_workers=WORKERS, pin_memory=PIN_MEMORY)
        
        model = PositionModel(input_dim=train_X.shape[1])
        
        print(f"Training {env} model...\n")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            patience=cfg["patience"],
            env_name=env,
            device=device
        )
        
        mae, rmse, status = evaluate_model(
            model=model,
            loader=test_loader,
            env_name=env,
            target_mae=cfg["target_mae"],
            device=device
        )
        
        summary[env] = {"MAE": mae, "RMSE": rmse, "Target": cfg["target_mae"], "Status": status}
    
    # Final summary
    print("\n" + "═"*80)
    print("FINAL RESULTS SUMMARY")
    print("═"*80)
    print(f"{'Environment':<12} {'MAE (m)':<10} {'RMSE (m)':<10} {'Target (m)':<12} {'Result'}")
    print("─"*80)
    for env, res in summary.items():
        print(f"{env:<12} {res['MAE']:<10.3f} {res['RMSE']:<10.3f} {res['Target']:<12} {res['Status']}")
    print("═"*80)

    print("\nDone. All environments processed.")