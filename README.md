## General Instructions :
1. If running on a local host: Ensure Python is present in your system and also see if these libraries are present in your system
   - numerical python [(numpy)](https://numpy.org/doc/stable/user/whatisnumpy.html)
   - weights and biases [(wandb)](https://docs.wandb.ai/?_gl=1*1lup0xs*_ga*NzgyNDk5ODQuMTcwNTU4MzMwNw..*_ga_JH1SJHJQXJ*MTcxMDY3NjQ2MS43Ny4xLjE3MTA2NzY0NjQuNTcuMC4w)
   - scikit-learn [(sklearn)](https://scikit-learn.org/stable/)
   - [matplotlib](https://matplotlib.org/)
   - [tensorflow](https://www.tensorflow.org/)
   - [keras](https://keras.io/guides/)
3. If running on colab/kaggle ignore point 1. It is suggested to run on your local machine to gain performance benefits.
4. Ensure that NeuralNetworks.py and train.py are present in the same directory.

follow this guide to install Python in your system:
1. Windows: https://kinsta.com/knowledgebase/install-python/#windows
2. Linux: https://kinsta.com/knowledgebase/install-python/#linux
3. MacOS: https://kinsta.com/knowledgebase/install-python/#mac

if the libraries are not present just run the command:
  pip install <library_name>

## Running the program:
Run the command(Runs in default settings mentioned in table below): 
``` python train.py ```

How to pass arguments:
``` python train.py -e 10 -lr 0.001 -cm 1```

**Arguments supported** :

| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| -d --dataset | fashion_mnist  |choices: ["mnist", "fashion_mnist"]|
|-e, --epochs|5|Number of epochs to train neural network.|
|-b, --batch_size|16|Batch size used to train neural network.|
|-l, --loss|cross_entropy|choices: ["mean_squared_error", "cross_entropy"]|
|-o, --optimizer	|nadam|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate|0.01|Learning rate used to optimize model parameters|
|-m, --momentum	|0.9|Momentum used by momentum and nag optimizers.|
|-beta, --beta	|0.9|Beta used by rmsprop optimizer|
|-beta1, --beta1|0.9|Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2|0.99|Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon|1e-10|Epsilon used by optimizers.|
|-w_d, --weight_decay|0.0005|	Weight decay/Regularization coefficient used by optimizers.|
|-w_i, --weight_init|Xavier|	choices: ["random", "Xavier"]|
|-nhl, --num_layers|5|Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size	|32|	Number of hidden neurons in a feedforward layer.|
|-a, --activation|tanh|	choices: ["identity", "sigmoid", "tanh", "ReLU"]|
|-cm, --conf_mat|default=0|Signifies whether you want to see the confusion matrix for the run|

## Additional support
### Defining a optimizer

Just pass your desired specification as command line argument, the rest is handled in the code
So, sit back and relax and observe :)

### Defining a new optimization algorithm
Open the **NeuralNetworks.py** file and open the **Gradient_descent class**:
just below the backpropagation algorithm copy this template
```python
def your_optimizer_algorithm_name(self):
   epochs = self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's
        
        I = Initializer()
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
        else:
            W,b = I.XavierIntializer(self.layers-1,self.npl)
        
        training_loss = []
        validation_loss = []
        training_accuracy = []
        validation_accuracy = []
        '''
            You can define rest of all your parameters here
            If you want to define parameters to same dimensions as W,b
            Option 1 : initialization to 0
                  W_like,b_like = I.Initialize(self.layers-1,self.npl)
            Option 2 : initialization to random
                  W_like,b_like = I.Initialize2(self.layers-1,self.npl)

            just copy and paste as you desire
            you can play around with your logic and define all of them accordingly
        '''
        for i in range(epochs):
            s = 0.0
            c = 0
            for j in range(0,len(x_train)//self.batch):
                
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1

                delW,delb = self.backward_propagation(A,H,W,b,y.T)
                PMA = Arithmetic()    # P - Plus, M - Minus , A - Arithmetic
                # if you want the ordinary updation rule like sgd,nag etc. :  W = PMA.Subtract(W,delW,self.eta)
                # if you want a updation rule like adam use this : W = PMA.AdamSubtract(W,mW_hat,vW_hat,eps,self.eta)
                W = PMA.Subtract(W,W,self.eta*self.config['regularization'])
                # if you want the ordinary updation rule like sgd,nag etc. : b = PMA.Subtract(b,delb,self.eta)
                # if you want a updation rule like adam use this : b = PMA.AdamSubtract(b,mb_hat,vb_hat,eps,self.eta)   
            
            training_loss.append(s/len(x_train))
            training_accuracy.append(c*100/len(x_train))
            vl = self.calc_val_loss(W,b)
            validation_loss.append(vl[0]/len(x_val))
            validation_accuracy.append(vl[1]*100/len(x_val))
            wandb.log({"training_loss": training_loss[-1],"validation_loss": validation_loss[-1],"training_accuracy":training_accuracy[-1],"validation_accuracy":validation_accuracy[-1],'epoch':i+1})
            print('******************************************************************************************')
            print('epoch : ',i+1)
            print('Training accuracy :',training_accuracy[-1],'Training Loss :',training_loss[-1]) 
            print('Validation accuracy :',validation_accuracy[-1],'Validation Loss :',validation_loss[-1])
            print('******************************************************************************************')
         
        return W,b
```
Now after adding your desired optimizer, add the following statement in the if-else-if block of the run_models function
```python
   elif self.config['optimizer'] == 'your_optimizer_algorithm_name' :
            W,b = self.your_optimizer_algorithm_name()
```
