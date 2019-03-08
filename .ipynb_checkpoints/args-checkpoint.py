import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='attention model parameters')
    # data
    parser.add_argument('--train', default='../../dir_HugeFiles/processed_data_0306/GI.pickle', type=str)
    parser.add_argument('--small', default=False, type=bool)

    # model
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--snapshots', default='snapshots/',type=str)
    parser.add_argument('--random', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=int)


    # training 
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--ingrW2V', default='../data/vocab.bin',type=str)
    parser.add_argument('--gloveW2V', default='../../dir_HugeFiles/glove.6B/glove.6B.100d.txt',type=str)
    parser.add_argument('--resume', default='', type=str)
    
    
    #gpu
    parser.add_argument('--gpu', default =2, type=int)
    #weight
    parser.add_argument('--pweight', default = -1, type = int)
    return parser




