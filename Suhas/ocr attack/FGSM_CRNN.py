import torch
import torchvision
from dataset import *
from utils import *
from PIL import Image
import models.crnn as crnn
import numpy as np 
import torch.optim as optim
from  torchvision import utils as vutils
# import matplotlib.pyplot as plt

use_cuda=True
model_path = './data/crnn.pth' 
# model_path = './data/netCRNN.pth' 
img_path = './data/oml.jpg' 
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

epsilon = [0, 0.01, 0.01, 0.01, 0.01]
# epsilon = 5

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = crnn.CRNN(32, 1, len(alphabet) + 1, 256).to(device)

# Load pre-train model
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))


converter = strLabelConverter(alphabet, ignore_case=False)


transformer = resizeNormalize((100, 32))

# FGSM attack 
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    image = image.squeeze(1)
    
    sign_data_grad = sign_data_grad.resize_([1,32,100])  
    sign_data_grad = sign_data_grad.sign() 
    print(image)
    perturbed_image = image + epsilon*sign_data_grad
    
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image   
    perturbed_image = perturbed_image.unsqueeze(1)
    # save perturbed_image
    vutils.save_image(perturbed_image, './data/adv_fgsm.png', normalize=True)
    # plt.imshow(perturbed_image, cmap="gray")
    # plt.savefig('./data/adv_fgsm.jpg')
    # plt.close()
    return perturbed_image

#100 x 32 w x h
image = Image.open(img_path).convert('L') 
image = transformer(image) # (1, 32, 100)
image = image.view(1, *image.size())  # (b, c, h, w) (1, 1, 32, 100)

model.eval()

criterion = torch.nn.CTCLoss()

def test( model, device, epsilon ):

    # Send the data to the device
    # target = [24, 11, 418, 323, 446, 2, 350, 335, 291, 109]
    data = image
    data = data.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    # data.requires_grad = True 
    # Forward pass the data through the model
    preds = model(data)
    preds = Variable(preds, requires_grad=True)
    
    
    batch_size = data.size(0)
    target, length = converter.encode(alphabet) 

    preds_size = torch.IntTensor([preds.size(0)] * batch_size)

    # Calculate the loss
    loss = criterion(preds, target, preds_size, length)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = preds.grad

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)

    # Re-classify the perturbed image
    output = model(perturbed_data)
    _, output = output.max(2) 
    output = output.transpose(1, 0).contiguous().view(-1)
    
    output_size = torch.IntTensor([output.size(0)])
    raw_output = converter.decode(output.data, output_size.data, raw=True)
    sim_output = converter.decode(output.data, output_size.data, raw=False)
    
    return raw_output, sim_output


# Run test for each epsilon
raw_output = []
sim_output = []
for eps in epsilon:
    raw_output, sim_output = test(model, device, eps)
    print("Epsilon: {}\t {} => {}".format(eps, raw_output, sim_output))