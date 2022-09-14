import numpy as np
import torch
from torch.autograd import Variable

from config import config






def train_multiclass(train_dataloader, model, optimizer, criterion, epoch):
    print('[INFO] training')
    model.train()
    counter = 0
    nProcessed = 0
    training_loss = 0.0
    running_corrects = 0

    for batch_idx, (x, y) in enumerate(train_dataloader):
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        y = y.float()

        optimizer.zero_grad()
        y_pred = model(x)
        y_pred = torch.softmax(y_pred, dim=1)

        y = torch.argmax(y, dim=1)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        nProcessed += config.batch_size
        counter += 1

        y_pred = torch.argmax(y_pred, dim=1)
        correct = torch.sum(y_pred == y.data)
        running_corrects += correct
        training_loss += loss.data.item() * x.size(0)
        acc = running_corrects / nProcessed

        print('Train Epoch: {}({}/{})]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, counter, len(train_dataloader), loss.item(), acc))



def test_multiclass(test_dataloader, model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            y = y.float()

            y_pred = model(x)
            y_pred = y_pred
            y_pred = torch.softmax(y_pred, dim=1)

            y = torch.argmax(y, dim=1)

            val_loss += criterion(y_pred, y)

            y_pred = torch.argmax(y_pred, dim=1)
            # y_pred = torch.round(y_pred)
            curr = torch.sum(y_pred == y.data)
            correct = correct + curr


    val_loss /= len(test_dataloader)  # loss function already averages over batch size
    nTotal = len(test_dataloader.dataset)

    val_acc = 100. * correct.item() / nTotal
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss.item(), correct, nTotal, val_acc))
    return val_loss, val_acc



def train_test(train_dataloader, validation_dataloader, model, optimizer, criterion, epoch):
    difficulty_arr = np.zeros(len(train_dataloader))

    train_multiclass(train_dataloader, model, optimizer, criterion, epoch)
    val_loss, val_acc = test_multiclass(validation_dataloader, model, criterion)

    if config.num_classes == 1:
        print('[WARNING] binary classification code has not been implemented in this section.')

    return val_loss, val_acc


def run(train_dataloader, validation_dataloader, model, optimizer, criterion, scheduler, epochs):
    validation_metric = 0
    weight_path = config.output_path + config.experiment_name + '/' + config.model_name + '/weights/'
    for epoch in range(0, epochs):
        val_loss, current_metric = train_test(train_dataloader, validation_dataloader, model, optimizer, criterion, epoch)
        scheduler.step(val_loss)
        if current_metric > validation_metric:
            validation_metric = current_metric
            torch.save(model, weight_path + '_epoch_' + str(epoch) + '_val_' + str(round(current_metric, 3)) + '_last_weights.pth')