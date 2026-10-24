import os
import json
import time
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.models as models
from torchvision.transforms import v2 as transforms
import kagglehub
import matplotlib.pyplot as plt

# pick gpu if we have one, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# grab the dataset from kagglehub and point to the test folder
path = kagglehub.dataset_download("desirewaheed/3d-body-mesh")
data_dir = os.path.join(path, "test")

# standard resnet transforms: resize, to tensor, normalize with imagenet stats
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# custom dataset that loads images and their pose json labels
class bodymeshdataset(Dataset):
    def __init__(self, root, transform=None):
        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")
        self.transform = transform
        # grab all image files, sorted so order is consistent
        self.filenames = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.classes = ["body_mesh"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # look for matching json label file, use zeros if missing
        base_name = os.path.splitext(name)[0]
        label_path = os.path.join(self.labels_dir, base_name + ".json")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                data = json.load(f)
                label_data = data["pose"]
                target = torch.tensor(label_data, dtype=torch.float32)
        else:
            target = torch.zeros(72, dtype=torch.float32)

        return image, target

# simple model: resnet34 backbone with a small head for 72 pose outputs
class posepredictionmodel(nn.Module):
    def __init__(self, output_dim=72):
        super(posepredictionmodel, self).__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        # replace the final fc layer with our own regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.backbone(x)

# load dataset and split into train/val/test (80/15/5)
dataset = bodymeshdataset(root=data_dir, transform=transform)
print(f"total images found: {len(dataset)}")
print(f"number of target classes: {len(dataset.classes)}")

total_size = len(dataset)
train_size = int(0.80 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"train size: {len(train_dataset)} | val size: {len(val_dataset)} | test size: {len(test_dataset)}")

# dataloaders with batch size 32
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# model, loss (mse for regression), and optimizer
model = posepredictionmodel(output_dim=72).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# training loop over 30 epochs
num_epochs = 30
train_losses = []
val_losses = []

print(f"starting tracking verification over {num_epochs} epochs...")
for epoch in range(num_epochs):
    start_time = time.time()

    # training phase
    model.train()
    running_train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # validation phase (no grad needed)
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_val_loss += loss.item() * images.size(0)

    epoch_val_loss = running_val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)

    epoch_time = time.time() - start_time

    print(f"epoch [{epoch+1:02d}/{num_epochs:02d}] | "
          f"train loss: {epoch_train_loss:.6f} | "
          f"val loss: {epoch_val_loss:.6f} | "
          f"time: {epoch_time:.1f}s")

print("done!!!")

# plot the loss curves so we can see how training went
completed_epochs = len(train_losses)
if completed_epochs > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, completed_epochs + 1), train_losses, label='training loss', color='blue', linewidth=2)
    plt.plot(range(1, completed_epochs + 1), val_losses, label='validation loss', color='orange', linewidth=2)

    plt.title('3d body mesh model - pose regression loss over epochs', fontsize=14, fontweight='bold')
    plt.xlabel('epochs', fontsize=12)
    plt.ylabel('loss (mean squared error)', fontsize=12)
    plt.xticks(range(1, completed_epochs + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.show()

# save just the model weights
model_save_path = "body_mesh_resnet34.pth"
torch.save(model.state_dict(), model_save_path)
print(f"model successfully saved to {model_save_path}!")

# also save a full checkpoint so we can resume training later if needed
checkpoint_path = "body_mesh_checkpoint.pth"
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses
}
torch.save(checkpoint, checkpoint_path)
print(f"full training checkpoint successfully saved to {checkpoint_path}!")