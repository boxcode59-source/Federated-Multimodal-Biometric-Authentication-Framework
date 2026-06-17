# ============================================================
# Federated Multimodal Biometric Authentication Framework
# using Cross-Modal Deep Feature Fusion and
# Privacy-Preserving Learning
# ============================================================

import os
import copy
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 106
BATCH_SIZE = 16
FEATURE_DIM = 256
NUM_CLIENTS = 5
LOCAL_EPOCHS = 2
GLOBAL_ROUNDS = 10
LR = 1e-4

# ============================================================
# DATASET
# ============================================================

class SDUMLADataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.samples = []

        # Expected Structure
        #
        # root/
        #    face/
        #    vein/
        #    gait/
        #    iris/
        #    fingerprint/
        #

        for cls in os.listdir(os.path.join(root_dir, "face")):

            cls_path = os.path.join(root_dir, "face", cls)

            if os.path.isdir(cls_path):

                for img in os.listdir(cls_path):

                    self.samples.append((cls, img))

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def load_img(self,path):

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img

    def __getitem__(self,idx):

        cls,img_name = self.samples[idx]

        label = int(cls)

        face = self.load_img(
            os.path.join(self.root_dir,"face",cls,img_name)
        )

        vein = self.load_img(
            os.path.join(self.root_dir,"vein",cls,img_name)
        )

        gait = self.load_img(
            os.path.join(self.root_dir,"gait",cls,img_name)
        )

        iris = self.load_img(
            os.path.join(self.root_dir,"iris",cls,img_name)
        )

        finger = self.load_img(
            os.path.join(self.root_dir,"fingerprint",cls,img_name)
        )

        return face,vein,gait,iris,finger,label

# ============================================================
# FACE FEATURE EXTRACTOR
# EfficientNet-B0
# ============================================================

class FaceNet(nn.Module):

    def __init__(self):

        super().__init__()

        net = models.efficientnet_b0(weights='DEFAULT')

        self.backbone = net.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1280,FEATURE_DIM)

    def forward(self,x):

        x = self.backbone(x)

        x = self.pool(x).flatten(1)

        x = self.fc(x)

        return x

# ============================================================
# FINGER VEIN
# DenseNet121
# ============================================================

class VeinNet(nn.Module):

    def __init__(self):

        super().__init__()

        net = models.densenet121(weights='DEFAULT')

        self.backbone = net.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1024,FEATURE_DIM)

    def forward(self,x):

        x = self.backbone(x)

        x = self.pool(x).flatten(1)

        x = self.fc(x)

        return x

# ============================================================
# GAIT CNN + BiLSTM
# ============================================================

class GaitNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(256,FEATURE_DIM)

    def forward(self,x):

        b = x.size(0)

        x = self.cnn(x)

        x = x.mean((2,3))

        x = x.unsqueeze(1)

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return out

# ============================================================
# IRIS RESNET50
# ============================================================

class IrisNet(nn.Module):

    def __init__(self):

        super().__init__()

        net = models.resnet50(weights='DEFAULT')

        net.fc = nn.Identity()

        self.backbone = net

        self.fc = nn.Linear(2048,FEATURE_DIM)

    def forward(self,x):

        x = self.backbone(x)

        x = self.fc(x)

        return x

# ============================================================
# FINGERPRINT MobileNetV3
# ============================================================

class FingerprintNet(nn.Module):

    def __init__(self):

        super().__init__()

        net = models.mobilenet_v3_large(weights='DEFAULT')

        self.features = net.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(960,FEATURE_DIM)

    def forward(self,x):

        x = self.features(x)

        x = self.pool(x).flatten(1)

        x = self.fc(x)

        return x

# ============================================================
# ACAF
# Adaptive Cross Modal Attention Fusion
# ============================================================

class ACAF(nn.Module):

    def __init__(self):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            FEATURE_DIM,
            num_heads=8,
            batch_first=True
        )

        self.fc = nn.Linear(
            FEATURE_DIM*5,
            FEATURE_DIM
        )

    def forward(self,features):

        x = torch.stack(features,dim=1)

        attn,_ = self.attn(x,x,x)

        attn = attn.reshape(attn.size(0),-1)

        out = self.fc(attn)

        return out

# ============================================================
# SPOOF DETECTION
# ============================================================

class SpoofDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(FEATURE_DIM,128),
            nn.ReLU(),

            nn.Linear(128,2)
        )

    def forward(self,x):

        return self.fc(x)

# ============================================================
# MAIN MODEL
# ============================================================

class FederatedBiometricModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.face = FaceNet()
        self.vein = VeinNet()
        self.gait = GaitNet()
        self.iris = IrisNet()
        self.finger = FingerprintNet()

        self.fusion = ACAF()

        self.auth_head = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES
        )

        self.spoof_head = SpoofDetector()

    def forward(self,
                face,
                vein,
                gait,
                iris,
                finger):

        f1 = self.face(face)
        f2 = self.vein(vein)
        f3 = self.gait(gait)
        f4 = self.iris(iris)
        f5 = self.finger(finger)

        fused = self.fusion(
            [f1,f2,f3,f4,f5]
        )

        auth = self.auth_head(fused)

        spoof = self.spoof_head(fused)

        return auth,spoof

# ============================================================
# FEDAVG
# ============================================================

def fedavg(models_list):

    global_model = copy.deepcopy(models_list[0])

    global_dict = global_model.state_dict()

    for k in global_dict.keys():

        global_dict[k] = torch.stack(
            [m.state_dict()[k].float()
             for m in models_list]
        ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model

# ============================================================
# LOCAL TRAIN
# ============================================================

def train_local(model,loader):

    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(LOCAL_EPOCHS):

        for face,vein,gait,iris,finger,label in loader:

            face = face.to(DEVICE)
            vein = vein.to(DEVICE)
            gait = gait.to(DEVICE)
            iris = iris.to(DEVICE)
            finger = finger.to(DEVICE)

            label = label.to(DEVICE)

            auth,spoof = model(
                face,
                vein,
                gait,
                iris,
                finger
            )

            loss = criterion(auth,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model,loader):

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():

        for face,vein,gait,iris,finger,label in loader:

            face = face.to(DEVICE)
            vein = vein.to(DEVICE)
            gait = gait.to(DEVICE)
            iris = iris.to(DEVICE)
            finger = finger.to(DEVICE)

            label = label.to(DEVICE)

            auth,_ = model(
                face,
                vein,
                gait,
                iris,
                finger
            )

            pred = auth.argmax(1)

            correct += (pred==label).sum().item()

            total += label.size(0)

    return 100*correct/total

# ============================================================
# FEDERATED TRAINING
# ============================================================

def federated_training(root):

    dataset = SDUMLADataset(root)

    client_size = len(dataset)//NUM_CLIENTS

    clients = torch.utils.data.random_split(
        dataset,
        [client_size]*(NUM_CLIENTS-1)
        + [len(dataset)-client_size*(NUM_CLIENTS-1)]
    )

    global_model = FederatedBiometricModel().to(DEVICE)

    for round_idx in range(GLOBAL_ROUNDS):

        local_models = []

        for client_data in clients:

            loader = DataLoader(
                client_data,
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            local_model = copy.deepcopy(
                global_model
            )

            local_model = train_local(
                local_model,
                loader
            )

            local_models.append(
                local_model
            )

        global_model = fedavg(
            local_models
        ).to(DEVICE)

        print(
            f"Global Round {round_idx+1} Completed"
        )

    return global_model

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATASET_PATH = "SDUMLA_HMT"

    global_model = federated_training(
        DATASET_PATH
    )

    torch.save(
        global_model.state_dict(),
        "Federated_Multimodal_Biometric.pth"
    )

    print("Training Completed")