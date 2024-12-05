
import torch
from yolov11 import YOLOv11  # Hypothetical import, replace with actual YOLO import

# Load the pre-trained YOLO model
model = YOLOv11(pretrained=True)

# Modify the model for the new number of classes
model.head.num_classes = 3  # Assuming 3 classes: person, truck, boat

# Load your custom dataset
from pycocotools.coco import COCO
import requests
import zipfile
import os

# Download the COCO dataset
coco_url = "http://images.cocodataset.org/zips/train2017.zip"
coco_zip = "train2017.zip"
coco_dir = "coco"

if not os.path.exists(coco_dir):
    os.makedirs(coco_dir)

if not os.path.exists(os.path.join(coco_dir, coco_zip)):
    print("Downloading COCO dataset...")
    r = requests.get(coco_url)
    with open(os.path.join(coco_dir, coco_zip), 'wb') as f:
        f.write(r.content)

    print("Extracting COCO dataset...")
    with zipfile.ZipFile(os.path.join(coco_dir, coco_zip), 'r') as zip_ref:
        zip_ref.extractall(coco_dir)

# Load the COCO dataset
coco = COCO(os.path.join(coco_dir, 'annotations', 'instances_train2017.json'))

# Filter the dataset for the specified classes
class_ids = coco.getCatIds(catNms=['person', 'truck', 'boat'])
img_ids = coco.getImgIds(catIds=class_ids)
train_dataset = torch.utils.data.Subset(CustomCocoDataset(root=coco_dir, classes=['person', 'truck', 'boat']), img_ids)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
# Define the loss function
def compute_loss(outputs, targets):
    # Assuming outputs and targets are dictionaries with keys 'boxes' and 'labels'
    # You can use a combination of classification and regression losses
    classification_loss = torch.nn.CrossEntropyLoss()(outputs['labels'], targets['labels'])
    regression_loss = torch.nn.SmoothL1Loss()(outputs['boxes'], targets['boxes'])
    
    # Combine the losses
    total_loss = classification_loss + regression_loss
    return total_loss

# Fine-tune the model
model.train()
for epoch in range(num_epochs):
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, targets)  # Define your loss function
        loss.backward()
        optimizer.step()



dummy_input = torch.randn(1, 3, 640, 640)  # Example input size
torch.onnx.export(model, dummy_input, "yolov11.onnx", opset_version=11)



import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("yolov11.onnx", "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))

builder.max_workspace_size = 1 << 30  # 1GB
builder.max_batch_size = 1
engine = builder.build_cuda_engine(network)

with open("yolov11.trt", "wb") as engine_file:
    engine_file.write(engine.serialize())