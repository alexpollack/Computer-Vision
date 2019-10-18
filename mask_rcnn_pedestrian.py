
# coding: utf-8

# In[1]:

# %%shell

# # Install pycocotools
# git clone https://github.com/cocodataset/cocoapi.git
# cd cocoapi/PythonAPI
# python setup.py build_ext install


# In[2]:

# %%shell

# # download the Penn-Fudan dataset
# wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# # extract it in the current folder
# unzip PennFudanPed.zip


# In[3]:

from PIL import Image
Image.open('PennFudanPed/PNGImages/FudanPed00001.png')


# In[4]:

mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
# each mask instance has a different color, from zero to N, where
# N is the number of instances. In order to make visualization easier,
# let's adda color palette to the mask.
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])
mask


# In[ ]:

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# In[6]:

dataset = PennFudanDataset('PennFudanPed/')
dataset[0]


# In[ ]:

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# In[8]:

# %%shell

# # Download TorchVision repo to use some files from
# # references/detection
# git clone https://github.com/pytorch/vision.git
# cd vision
# git checkout v0.3.0

# cp references/detection/utils.py ../
# cp references/detection/transforms.py ../
# cp references/detection/coco_eval.py ../
# cp references/detection/engine.py ../
# cp references/detection/coco_utils.py ../


# In[ ]:

from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# In[ ]:

# use our dataset and defined transformations
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# In[11]:

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# ## Train for 10 epochs

# In[13]:

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
#     evaluate(model, data_loader_test, device=device)


# # Test image 1

# In[ ]:

# pick one image from the test set
test1, _ = dataset_test[10]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction1 = model([test1.to(device)])


# In[21]:

prediction1


# In[22]:

Image.fromarray(test1.mul(255).permute(1, 2, 0).byte().numpy())


# In[23]:

Image.fromarray(prediction1[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 2

# In[ ]:

# pick one image from the test set
test2, _ = dataset_test[5]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction2 = model([test2.to(device)])


# In[32]:

Image.fromarray(test2.mul(255).permute(1, 2, 0).byte().numpy())


# In[33]:

Image.fromarray(prediction2[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 3

# In[ ]:

# pick one image from the test set
test3, _ = dataset_test[4]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction3 = model([test3.to(device)])


# In[36]:

Image.fromarray(test3.mul(255).permute(1, 2, 0).byte().numpy())


# In[37]:

Image.fromarray(prediction3[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 4

# In[ ]:

# pick one image from the test set
test4, _ = dataset_test[9]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction4 = model([test4.to(device)])


# In[39]:

Image.fromarray(test4.mul(255).permute(1, 2, 0).byte().numpy())


# In[40]:

Image.fromarray(prediction4[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 5

# In[ ]:

# pick one image from the test set
test5, _ = dataset_test[3]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction5 = model([test5.to(device)])


# In[42]:

Image.fromarray(test5.mul(255).permute(1, 2, 0).byte().numpy())


# In[43]:

Image.fromarray(prediction5[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 6

# In[ ]:

# pick one image from the test set
test6, _ = dataset_test[7]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction6 = model([test6.to(device)])


# In[51]:

Image.fromarray(test6.mul(255).permute(1, 2, 0).byte().numpy())


# In[52]:

Image.fromarray(prediction6[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 7

# In[ ]:

# pick one image from the test set
test7, _ = dataset_test[11]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction7 = model([test7.to(device)])


# In[54]:

Image.fromarray(test7.mul(255).permute(1, 2, 0).byte().numpy())


# In[55]:

Image.fromarray(prediction7[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 8

# In[ ]:

# pick one image from the test set
test8, _ = dataset_test[13]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction8 = model([test8.to(device)])


# In[57]:

Image.fromarray(test8.mul(255).permute(1, 2, 0).byte().numpy())


# In[58]:

Image.fromarray(prediction8[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 9

# In[ ]:

# pick one image from the test set
test9, _ = dataset_test[4]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction9 = model([test9.to(device)])


# In[60]:

Image.fromarray(test9.mul(255).permute(1, 2, 0).byte().numpy())


# In[61]:

Image.fromarray(prediction9[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# # Test image 10

# In[ ]:

# pick one image from the test set
test10, _ = dataset_test[15]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction10 = model([test10.to(device)])


# In[63]:

Image.fromarray(test10.mul(255).permute(1, 2, 0).byte().numpy())


# In[64]:

Image.fromarray(prediction10[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


# In[ ]:



