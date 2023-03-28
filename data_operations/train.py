import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm

from config import config
from data_operations.train_test_loops import *
from data_operations.load_data import *
from data_operations.write_params import *
from data_operations.train_operations import *
from callbacks.make_experiment import *

from models.s_gnn_similarity import *

def perform_training():
    torch.cuda.manual_seed(1776)
    cudnn.benchmark = True
    make_experiment()
    dataproc = data_preprocess(config)

    train_dataset = ImageDataset(dataproc.train_imgs, dataproc.train_gt, config.input_dim, config.augment_train)
    val_dataset = ImageDataset(dataproc.val_imgs, dataproc.val_gt, config.input_dim, config.augment_validation)



    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)





    model = config.model


    if config.resume_training:
        model = torch.load(config.resume_training_path)
        print('[INFO] resuming training from previously stored weights')

    model = model.cuda()

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum) #weight_decay=config.weight_decay
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, patience=config.lr_patience)

    if config.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        print('[ERROR] No loss function has been selected'); quit()

    write_data()
    run(train_dataloader,
        validation_dataloader,
        model,
        optimizer,
        criterion,
        scheduler,
        config.epochs)


if __name__ == '__main__':
    perform_training()