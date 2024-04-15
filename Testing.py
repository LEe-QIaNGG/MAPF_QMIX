import Env
import DQN
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image
from pix2tex.cli import LatexOCR

# img = Image.open("F:/课程/7毕设/pic/eq.png")
# model = LatexOCR()
# print(model(img))


print(torch.__version__)
print(torch.cuda.is_available())


