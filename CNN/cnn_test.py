import torch
from cnn_train import Loader, Net, ModelTrainer


if __name__ == '__main__':
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = Loader(partition='test')
    
    cnn_model = Net()

    # create tester
    tester = ModelTrainer(model=cnn_model,
                           data_loader=train_loader)

    # initialize gnn pre-trained
    checkpoint = torch.load('./trained_cnn_model.pth')
    tester.model.load_state_dict(checkpoint)
    print("load pre-trained cnn done!")

    tester.test()
