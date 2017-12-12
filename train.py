import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt

from model import Model
from data import Env2DDataset, PositionDataset, ThorDataset


DIM = 2
CHANNELS = 1
OUTPUT_SIZE = 2
EPOCHS = 1000
LAMBDA = 0.1

reg_scale = 0.002
rate = 1.0
printEpoch = 0

def directionVal(labels, predictions, a):
	#labels = labels.numpy()
	C = np.multiply(labels, predictions)
	C = np.array([x[0] for x in C])
	D = np.ndarray.flatten(C)
	
	n = float(len(D))
	G = (D >= 0.0).sum()
	s = ["train", "validation"]
	print(s[a], " correct direction percentage:", G, "/", n, "=", G / n)

	


def train(model, train_dataset, val_dataloader, optimizer,
          position_criterion, prediction_criterion):
    loss = 0
    
    best_val_loss = 1e5
    best_val_epoch = 0
    train_loss_hist = []
    val_loss_hist = []
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    for epoch in range(EPOCHS):
        
        print('Epoch', epoch + 1)
        for data in train_dataloader:
            model.train()
            labels = data['position']
            labels = Variable(labels.float())
            frames = data['frames']
            #print(frames[0].numpy().shape)
            frames = [Variable(f.float()) for f in frames]
            input_frames = frames[0], frames[1]
            
            pred_frame = frames[1]

            optimizer.zero_grad()
            positions, prediction = model(input_frames)
 #           print(positions.data.numpy()[0], labels.data.numpy()[0])
            position_loss = position_criterion(positions, labels)
            directionVal(labels.data.numpy(), positions.data.numpy(),0)
            #prediction_loss = prediction_criterion(prediction, pred_frame)
            #loss = (1 - LAMBDA) * position_loss + LAMBDA * prediction_loss
            loss = position_loss
            #print(loss)
            l2_reg = None
            for W in model.parameters():
            	if l2_reg is None:
            		l2_reg = 2 * W.norm(2)  + W.norm(1)
            else:
            	l2_reg = l2_reg + 2 * W.norm(2) + W.norm(1)
            loss = rate * loss + reg_scale * l2_reg
            loss.backward()
            optimizer.step()
            #train_loss_hist.append((loss.data[0], position_loss.data[0], prediction_loss.data[0]))
        if (epoch > printEpoch):
            train_loss_hist.append((np.mean(loss.data.numpy()), np.mean(position_loss.data.numpy()), 0))
            val_loss_hist.append((-1, -1, -1))
        # print('{} / {}'.format(prediction, frames[2]))
        # print('{} / {}'.format(positions, labels))
        print('Losses:', loss.data[0], position_loss.data[0],
              #prediction_loss.data[0])
              0)

        val_loss = 0
        for data in val_dataloader:
            model.eval()
            frames = data['frames']
            frames = [Variable(f.float(), volatile=True) for f in frames]
            input_frames = frames[0], frames[1]
            pred_frame = frames[1]
            positions, prediction = model(input_frames)
            position_loss = position_criterion(positions, Variable(data['position'].float()))
            #prediction_loss = prediction_criterion(prediction, pred_frame)
            #val_loss += (1 - LAMBDA) * position_loss + LAMBDA * prediction_loss
            val_loss = rate * position_loss + reg_scale * l2_reg
            #loss = loss - val_loss
            #loss.backward()
            #optimizer.step()

            directionVal(data['position'].float().numpy(), positions.data.numpy(), 1)
        #val_loss_hist[-1] = (val_loss.data[0], position_loss.data[0], prediction_loss.data[0])

        if (epoch > printEpoch):
        	mean = np.mean(position_loss.data.numpy())
        	val_loss_hist[-1] = ((np.mean(val_loss.data.numpy()), mean, 0))
        	if mean < best_val_loss:
        	    model.save('best_val_model')
        	    best_val_loss = mean
        	    best_val_epoch = epoch + 1
        	    print('Saving model at epoch', best_val_epoch)
        	print('Validation losses:', val_loss.data[0], position_loss.data[0],
        	      #prediction_loss.data[0])
        	      0)

        	plt.clf()
        	train_x_axis = range(len(train_loss_hist))
        	val_x_axis = [train_x_axis[i] for i, x in enumerate(val_loss_hist) if x[0] >= 0]
        	plt.plot(train_x_axis, [p[1] for p in train_loss_hist],
        	         label='train loss')
        	# plt.plot(train_x_axis, [p[1] for p in train_loss_hist],
        	#          label='train pos loss')
        	# plt.plot(train_x_axis, [p[2] for p in train_loss_hist],
        	#          label='train pred loss')
        	plt.plot(val_x_axis, [p[1] for p in val_loss_hist if p[0] >= 0],
        	         label='val loss')
        	# plt.plot(val_x_axis, [p[1] for p in val_loss_hist if p[1] >= 0],
        	#          label='val pos loss')
        	# plt.plot(val_x_axis, [p[2] for p in val_loss_hist if p[2] >= 0],
        	#          label='val pred loss')
        	plt.legend(loc='center left')
        	plt.pause(0.0001)

    print('Lowest validation loss:', best_val_loss,
          'at epoch', best_val_epoch)
    plt.show()


def test(model, test_dataloader, position_criterion):
    print('Loading best validation parameters...')
    model.load('best_val_model')
    model.eval()
    position_loss = 0
    for data in test_dataloader:
        frames = data['frames'][:-1]
        frames = [Variable(f.float(), volatile=True) for f in frames]
        positions, _ = model(frames)

        position_loss += position_criterion(positions, Variable(data['position'].float())).data[0]
    # print(torch.cat([positions.data.t(),
    #                  data['position'].t().float()]))
    print(position_loss)


if __name__ == '__main__':
    plt.ion()
    model = Model(DIM, CHANNELS, OUTPUT_SIZE)
    model.load('best_val_model')
    transform = None
    # transform = transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)
    #train_dataset = Env2DDataset("env_2d/train/data.txt", transform=transform)
    train_dataset = ThorDataset("Thor/out", transform=transform)
    

    #val_dataset = Env2DDataset("env_2d/val/data.txt", transform=transform)
    print("loading dataset")
    val_dataset = ThorDataset("Thor/val", transform=None)
    print("loading dataset - 2")
    val_dataloader = DataLoader(val_dataset, batch_size=50)

    position_criterion = nn.MSELoss()
    prediction_criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0005)

    train(model, train_dataset, val_dataloader, optimizer, position_criterion, prediction_criterion)

    #test_dataset = Env2DDataset("env_2d/test/data.txt", transform=transform)
    #test_dataloader = DataLoader(test_dataset, batch_size=500)
    #test(model, test_dataloader, position_criterion)
