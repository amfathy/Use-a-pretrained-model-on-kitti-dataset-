import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch_yolov3 import yolov3

# Define your custom dataset
class KITTIDataset(Dataset):
    def __init__(self, root, transforms=None, split='training'):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, f"{split}/image_2"))))
        self.split = split

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, f"{self.split}/image_2", self.imgs[idx])

        # Open the image file
        img = Image.open(img_path).convert("RGB")

        # Generate dummy annotations (as we don't have the label_2 directory)
        boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss {losses.item()}")

# Simplified evaluation function to calculate accuracy
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                pred_labels = output['labels']
                true_boxes = targets[i]['boxes'].to(device)
                true_labels = targets[i]['labels'].to(device)
                # Simplified accuracy calculation (needs proper IoU and matching logic)
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    correct += 1 if torch.allclose(pred_boxes[0], true_boxes[0], atol=10) else 0
                total += 1
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == '__main__':
    # Define dataset and dataloaders
    data_dir = 'D:/Collegematrial/Machinevision/Assignment5/data_object_image_2'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = KITTIDataset(root=data_dir, transforms=transform, split='training')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Define train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Print sample data to verify
    for images, targets in train_loader:
        print(images)
        print(targets)
        break

    # Load the YOLOv3 model pretrained on COCO dataset
    model = yolov3(pretrained=True)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        # Evaluate every epoch to track progress
        evaluate(model, test_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'yolov3_kitti.pth')

    # Evaluate the model
    train_accuracy = evaluate(model, train_loader, device)
    test_accuracy = evaluate(model, test_loader, device)

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
