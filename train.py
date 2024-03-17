import NeuralNetworks as NN
import argparse
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

def main(args):

    config['dataset'] = args.dataset
    config['activation'] = args.activation
    config['batch_size'] = args.batch_size
    config['Initialization'] = args.weight_init
    config['beta1'] = args.beta1
    config['beta2'] = args.beta2
    config['epochs'] = args.epochs
    config['neurons_per_layer'] = args.hidden_size
    config['layers'] = args.num_layers + 1
    config['loss'] = args.loss
    config['optimizer'] = args.optimizer
    config['learning_rate'] = args.learning_rate
    config['regularization'] = args.weight_decay
    config['epsilon'] = args.epsilon

    if(config['optimizer']=='rmsprop'):
        config['beta'] = args.beta
    else:
        config['beta'] = args.momentum

    wandb.init(project=args.wandb_project)
    #config = wandb.config
    
    model = NN.Gradient_descent(784,10,config)
    W,b = model.Run_Models()
    
    if args.conf_mat == 1:
        predict = NN.Prediction()
        al,pl = predict.predict_test(W,b,config)
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(y_true=np.argmax(al,axis=1),preds=np.argmax(pl,axis=1))})
    
    wandb.finish()

config = {
    'dataset' : 'fashion_mnist',
    'optimizer': 'nadam',
    'epochs': 5,
    'activation': 'tanh',
    'loss': 'cross_entropy',
    'layers': 5,
    'neurons_per_layer': 32,
    'learning_rate' : 0.1,
    'batch_size': 32,
    'regularization' : 0.0005,
    'beta': 0.9,
    'beta1': 0.9,
    'beta2': 0.999,
    'Initialization': 'Xavier',
    'epsilon' : 0.000001
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep_LearingAssignment1_CS23M062 -command line arguments")
    parser.add_argument("-wp","--wandb_project", type=str, default ='Shubhodeep_Final_CS6190_DeepLearing_Assignment1', help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we","--wandb_entity", type=str, default ='shubhodeepiitm062',help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d","--dataset",type=str,default ='fashion_mnist',help="dataset choices: [mnist, fashion_mnist]")
    parser.add_argument("-e","--epochs",type=int,default = 20,help ='Number of epochs to train neural network.')
    parser.add_argument("-b","--batch_size",type=int,default = 64,help='Batch size used to train neural network.')
    parser.add_argument("-l","--loss",type=str,default='cross_entropy',help= 'choices: ["mean_squared_error", "cross_entropy"]')
    parser.add_argument('-o','--optimizer',type=str,default='nadam',help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-lr','--learning_rate',type=float,default=0.01,help='Learning rate used to optimize model parameters')
    parser.add_argument('-m','--momentum',type=float,default=0.5,help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta','--beta',type = float,default=0.9,help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1','--beta1',type=float,default=0.9,help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2','--beta2',type=float,default=0.999,help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps','--epsilon',type=float,default=1e-10,help='Epsilon used by optimizers.')
    parser.add_argument('-w_d','--weight_decay',type=float,default=0.0005,help='Weight decay used by optimizers.')
    parser.add_argument('-w_i','--weight_init',type=str,default='Xavier',help='Initialization choices: ["random", "Xavier"]')
    parser.add_argument('-nhl','--num_layers',type=int,default=5,help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz','--hidden_size',type=int,default=256,help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a','--activation',type=str,default='tanh',help = 'choices: ["identity", "sigmoid", "tanh", "ReLU"]')
    parser.add_argument('-cm','--conf_mat',type=int,default=0,help='Signifies whether you want to see the confusion matrix for the run')
    args = parser.parse_args()
    main(args)



'''
sweep_config = {
    'method': 'bayes',
    'name' : 'sweep cross entropy',
    'metric': {
      'name': 'validation_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'dataset' : {'values' : ['fashion_mnist']},
        'optimizer' : {'values' : ['nadam']},
        'epochs': {'values': [5]},
        'activation': {'values': ['tanh']},
        'loss': {'values': ['cross_entropy']},
        'layers': {'values' : [3]},
        'neurons_per_layer' : {'values' : [32]},
        'learning_rate' : {'values' : [0.01]},
        'batch_size' : {'values' : [16]},
        'regularization' : {'values': [0]},
        'beta' : {'values' : [0.5]},
        'beta1' : {'values' : [0.9]},
        'beta2' : {'values' : [0.999]},
        "Initialization" : {'values' :['Xavier']}
    }
}
sweep_id = wandb.sweep(sweep=sweep_config,project="Shubhodeep_Final_CS6190_DeepLearing_Assignment1")
wandb.agent(sweep_id, function=main,count=1) # calls main function for count number of times.
wandb.finish()
'''
