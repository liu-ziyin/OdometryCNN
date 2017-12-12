import os

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import Env2DDataset, PositionDataset, ThorDataset
from model import Model
from train import directionVal, DIM, CHANNELS, OUTPUT_SIZE

if __name__ == '__main__':
    model = Model(DIM, CHANNELS, OUTPUT_SIZE)

    model_name = "best_val_model_fourier"
    dataset = "val_fft_binary"

    print("loading model %s..." % model_name)
    model.load(model_name)

    dataset_dir = os.path.join("Thor", dataset)
    print("loading validation dataset from %s..." % dataset_dir)
    val_dataset = ThorDataset(dataset_dir, transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=9999)

    position_criterion = nn.MSELoss()

    print("testing...")
    for data in val_dataloader:
        model.eval()
        frames = data['frames']
        frames = [Variable(f.float(), volatile=True) for f in frames]
        input_frames = frames[0], (frames[1] - frames[0])

        positions, prediction = model(input_frames)
        position_loss = position_criterion(positions, Variable(data['position'].float()))
        print("Loss is", position_loss.data.numpy())
        directionVal(data['position'].float().numpy(), positions.data.numpy(), 1)
