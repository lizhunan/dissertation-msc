# import pickle

# nyutest=open('src/datasets/sunrgbd/weighting_test.pickle','rb')
# data3=pickle.load(nyutest)
# print(data3)
# nyutest.close()

import re
import matplotlib.pyplot as plt

with open("slurm-997436.out", "r") as f:
    lines = f.readlines()

epochs = []
train_losses = []
val_losses = []

for line in lines:
    match = re.search(r"Epoch\s+(\d+).*Loss: (\d+\.\d+)", line)
    if match:
        epoch = int(match.group(1))
        loss = float(match.group(2))

        if epoch <=250:
            if "Train" in line:
                print(f'{epoch},{loss}')
                train_losses.append(loss)
                epochs.append(epoch) 
            else:
                val_losses.append(loss)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, '-o', label="Train Loss", color="blue")
plt.plot(epochs, val_losses, '-o', label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_vs_epoch.png", dpi=300)
