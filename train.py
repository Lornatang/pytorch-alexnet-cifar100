# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import random
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader

from nets.alexnet_test import alexnet
from utils.adjust import adjust_learning_rate
from utils.datasets import load_datasets
from utils.eval import accuracy
from utils.misc import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch CIFAR Classifier')
parser.add_argument('--dataroot', type=str, default="~/pytorch_datasets", help="download train dataset path.")
parser.add_argument('--datasets', type=str, default="cifar10", help="cifar10/cifar100 datasets. default=`cifar10`")
parser.add_argument('--batch_size', type=int, default=128, help="Every train dataset size.")
parser.add_argument('--lr', type=float, default=0.01, help="starting lr, every 10 epoch decay 10.")
parser.add_argument('--momentum', type=float, default=0.9, help="The ratio of accelerating convergence.")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Mainly to prevent overfitting.")
parser.add_argument('--epochs', type=int, default=50, help="Train loop")
parser.add_argument('--every_epoch', type=int, default=10, help="Every epoch lr / 10.")
parser.add_argument('--phase', type=str, default='eval', help="train or eval?")
parser.add_argument('--model_path', type=str, default="", help="load model path.")
opt = parser.parse_args()
print(opt)

try:
  os.makedirs("./checkpoints")
except OSError:
  pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataloader, test_dataloader = load_datasets(opt.datasets, opt.dataroot, opt.batch_size)

# Load model
if opt.datasets == "cifar100":
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(alexnet())
  else:
    model = alexnet()
else:
  model = ""
  print(opt)

model.to(device)
print(model)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


def train(train_dataloader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, data in enumerate(train_dataloader):

    # measure data loading time
    data_time.update(time.time() - end)

    # get the inputs; data is a list of [inputs, labels]
    inputs, targets = data
    inputs = inputs.to(device)
    targets = targets.to(device)

    # compute output
    output = model(inputs)
    loss = criterion(output, targets)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output, targets, topk=(1, 5))
    losses.update(loss.item(), inputs.size(0))
    top1.update(prec1, inputs.size(0))
    top5.update(prec5, inputs.size(0))

    # compute gradients in a backward pass
    optimizer.zero_grad()
    loss.backward()

    # Call step of optimizer to update model params
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 5 == 0:
      print(f"Epoch [{epoch + 1}] [{i}/{len(train_dataloader)}]\t"
            f"Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
            f"Loss {loss.item():.4f}\t"
            f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
            f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})", end="\r")
  torch.save(model.state_dict(), f"./checkpoints/{opt.datasets}_epoch_{epoch + 1}.pth")


def test(model):
  # switch to evaluate mode
  model.eval()
  # init value
  total = 0.
  correct = 0.
  with torch.no_grad():
    for i, data in enumerate(test_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()

  accuracy = 100 * correct / total
  return accuracy

def run():
  best_prec1 = 0.
  for epoch in range(opt.epochs):
    # Adjust learning rate according to schedule
    adjust_learning_rate(opt.lr, optimizer, epoch, opt.every_epoch)

    # train for one epoch
    print(f"\nBegin Training Epoch {epoch + 1}")
    train(train_dataloader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    print(f"Begin Validation @ Epoch {epoch + 1}")
    prec1 = test(model)

    # remember best prec@1 and save checkpoint if desired
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print(f"\tEpoch Accuracy: {prec1}")
    print(f"\tBest Accuracy: {best_prec1}")


if __name__ == '__main__':
  if opt.phase == "train":
    run()
  elif opt.phase == "eval":
    if opt.model_path != "":
      print("Loading model...\n")
      model.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))
      print("Loading model successful!")
      accuracy = test(model)
      print(f"\nAccuracy of the network on the 10000 test images: {accuracy:.2f}%.\n")
      visual(model)
    else:
      print("WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
  else:
    print(opt)
