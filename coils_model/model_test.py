import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import data_processor, FC_model

if __name__ == '__main__':
    state_path = "saved_models/coils_model_state_dict.pt"
    net_dims=[6, 32, 64, 64, 32, 2]
    batch_size: int = 64
    
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Model file not found: {state_path}")
    
    train_ds, val_ds, test_ds, meta = data_processor.load_data(val_ratio=0.0, test_ratio=0.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FC_model.FullyConnectedNet(net_dims)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    
    L_per_sample_errs = []
    R_per_sample_errs = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            rel = (preds - yb).abs() / yb.abs()
            L_per_sample = rel[:, 0]
            R_per_sample = rel[:, 1]
            L_per_sample_errs.append(L_per_sample.cpu().numpy())
            R_per_sample_errs.append(R_per_sample.cpu().numpy())

    L_per_sample_errs = np.concatenate(L_per_sample_errs, axis=0)
    R_per_sample_errs = np.concatenate(R_per_sample_errs, axis=0)

    fig = plt.figure(figsize=(12, 9))
    
    # 分别绘制电感和电阻的预测误差
    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(0, len(L_per_sample_errs)), 
        list(map(lambda x: x*100, L_per_sample_errs)),
        color="#0A14DA")
    ax1.set_xlim([0, len(L_per_sample_errs)])
    ax1.grid(True)
    ax1.set_xlabel('sample index')
    ax1.set_ylabel('Relevant Error (%)')
    ax1.set_title("Model's Relevant Error on Inductance (L) Prediction")

    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(0, len(R_per_sample_errs)), 
        list(map(lambda x: x*100, R_per_sample_errs)),
        color="#DA0A0A")
    ax2.set_xlim([0, len(R_per_sample_errs)])
    ax2.grid(True)
    ax2.set_xlabel('sample index')
    ax2.set_ylabel('Relevant Error (%)')
    ax2.set_title("Model's Relevant Error on Resistance (R) Prediction")

    plt.tight_layout()
    plt.show()


