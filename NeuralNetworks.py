'''
    Importing all the necessary libraries
'''

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import fashion_mnist,mnist
import math

wandb.login()

# Load fashion_MNIST data using Keras
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train,x_val, y_train, y_val=train_test_split(x_train,y_train, test_size=0.2,shuffle=True,random_state=42)

'''
    class Activation :  contains all the necessary functions required for the artificial neurons
    sigmoid -> sigmoid activation function
    g3 -> relu activation function value_returned = max(a,0)
    softmax -> function for the final y_hat, this returns a probability distribution over all classes
'''

class Activations :
    def sigmoid(self,x) :
        x = np.clip(x,-200,200)
        return 1/(1 + np.exp(-x))
    
    def g3(self,a):
        return np.maximum(a,0)

    def SoftMax(self,a):
        max_a = np.max(a)
        exp_a = np.exp(a - max_a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

'''
    class Differential :  contains all the necessary functions required for the artificial neurons to find the derivative of a function
    sig_dif -> differential function for sigmoid neuron
    tan_dif -> differential function for tanh neuron
    Rel_dif -> differential function for relu neuron
    Iden_dif -> differential function for identity neuron
'''

class Differential :
    def sig_dif(self,a):
        # if f(x) = 1/1+e^(-x)
        # the f_dash(x) = f(x)*(1-f(x)), this is what is implemented here
        activ = Activations()
        g_x = activ.sigmoid(a)
        return g_x*(1-g_x)

    def tan_dif(self,a):
        # if f(x) = e^x - e^-x/e^x + e^-x
        # the f_dash(x) = (1 - f(x)^2), this is what is implemented here
        g_dash = np.tanh(a)
        return 1 - (g_dash**2)  

    def Rel_dif(self,a):
        #       if value inside entries of a>0 then set to true else set to false, 
        #       .astype('float64') converts true/false into 1/0
        return (a > 0).astype('float64')        
    
    def Iden_dif(self,a):
        # if f(x) = x
        # the f_dash(x) = 1, this is what is implemented here
        g_dash = a
        g_dash[:] = 1
        return g_dash

class Initializer :

    def Initialize(self,hidden_layers,npl):
        W = [[]]    # list consisting of all the W's

        # Create and append each Wi matrix filled with zeros to the list
        W.append(np.zeros((npl,784)))           #input layer h0 weight  : W1

        for _ in range(hidden_layers-1):
            W.append(np.zeros((npl,npl)))       #create the W for each hidden layer : [ W2 ... W(L-1) ]

        W.append(np.zeros((10,npl)))            #Weight for the final layer : WL

        b = [[]]    # list consiting of all the b's

        # Create and append each bi vector filled with zeros to the list
        b.append(np.zeros(npl))       #input layer h0 bias

        for _ in range(hidden_layers-1):
            b.append(np.zeros(npl))   #create the b for each hidden layer : [ b2 ... b(L-1) ]

        b.append(np.zeros(10))        #bias for the final layer : bL

        return W,b


    def Initialize2(self,hidden_layers,npl):
        W = [[]]    # list consisting of all the W's

        # Create and append each Wi matrix filled with random values to the list
        W.append(np.random.randn(npl, 784))         #input layer h0 weight  : W1
        for _ in range(hidden_layers-1):
            W.append(np.random.randn(npl, npl))     #create the W for each hidden layer : [ W2 ... W(L-1) ]

        W.append(np.random.randn(10, npl))      #Weight for the final layer : WL

        b = [[]]    # list consiting of all the b's

        # Create and append each bi vector filled with random values to the list
        b.append(np.random.randn(npl))          #input layer h0 bias

        for _ in range(hidden_layers-1):
            b.append(np.random.randn(npl))      #create the b for each hidden layer : [ b2 ... b(L-1) ]
                
        b.append(np.random.randn(10))       #bias for the final layer : bL
        return W,b

    def XavierIntializer(self,hidden_layers,npl):
        W = [[]]  # list consisting of all the W's

        # Xavier initialization for weights
        def xavier_init(n_in, n_out):
            return np.random.randn(n_out, n_in)* np.sqrt(1 / n_in)      

        # Create and append each Wi matrix filled with Xavier-initialized values to the list
        W.append(xavier_init(784, npl))  # input layer h0
        for _ in range(hidden_layers - 1):
            W.append(xavier_init(npl, npl))

        W.append(xavier_init(npl, 10))

        b = [[]]  # list consisting of all the b's

        # Xavier initialization for biases (can also be initialized with zeros)
        def bias_init(n_out):
            return np.zeros(n_out, dtype=np.float64)

        # Create and append each bi vector filled with Xavier-initialized values to the list
        b.append(bias_init(npl))
        for _ in range(hidden_layers - 1):
            b.append(bias_init(npl))

        b.append(bias_init(10))
        return W, b

class Arithmetic :
    def Add(self,u,v):          #Adds two matrices u,v of the same order
        for i in range(1,len(u)):
            u[i] = u[i] + v[i]
        return u

    def Subtract(self,v,dv,eta):    #Update rule : W(t+1) = W(t) + eta*delW
        for i in range(1,len(v)):
            v[i] = v[i] - (eta * dv[i])
        return v

    def RMSpropSubtract(self,v,dv,lv,eps,eta):     #Update rule : W(t+1) = W(t) + (eta/(sqrt(v) + Epsilon)*delW
        for i in range(1,len(v)):
            ueta = eta/(np.sqrt(np.sum(lv[i])) + eps)
            v[i] = v[i] - (ueta * dv[i])
        return v

    def AdamSubtract(self,V,mV_hat,vV_hat,eps,eta):     #Update rule : W(t+1) = W(t) + (eta/(sqrt(v) + Epsilon)*delW
        for i in range(1,len(V)):
            norm = np.linalg.norm(vV_hat[i])
            ueta = eta/(np.sqrt(norm) + eps)
            V[i] = V[i] - (ueta * mV_hat[i])
        return V


class Gradient_descent :
    #constructor to create a neural network as specified by the user

    def __init__(self, input_size, output_size, config,flag = 0):
        global x_test,x_train,x_val,y_test,y_train,y_val
        self.input_size = input_size
        self.output_size = output_size
        self.layers = config['layers']
        self.activation = config['activation']
        self.npl = config["neurons_per_layer"]
        self.eta = config["learning_rate"]
        self.batch = config["batch_size"]
        self.init = config["Initialization"]
        self.config = config

        if not flag:
            if(config['dataset']=='mnist'):
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                x_train,x_val, y_train, y_val=train_test_split(x_train,y_train, test_size=0.2,shuffle=True,random_state=42)
            
            
            # Preprocess the data
            x_train = x_train/255.0 #.astype('float128') / 255.0
            x_val = x_val/255.0 #astype('float128')/ 255.0
            x_test = x_test/255.0

            encoder = OneHotEncoder(sparse_output=False)

            # Fit and transform the target values using OneHotEncoder
            y_train = encoder.fit_transform(y_train.reshape(-1, 1))
            #print(y_train[0])
            y_val = encoder.transform(y_val.reshape(-1, 1))
            y_test = encoder.transform(y_test.reshape(-1, 1))
            #print(y_train[0],y_val[0],y_test[0])

            # Flatten the images
            x_train = x_train.reshape((-1, 28 * 28))
            #print(x_train.shape)
            x_val = x_val.reshape((-1, 28 * 28))
            x_test = x_test.reshape((-1, 28 * 28))

    def backward_propagation(self,A,H,W,b,y):
        # Get the number of layers
        L = self.layers

        # Initialize lists to store gradients
        delA = [[]]*(L+1)
        delW = [[]]*(L+1)
        delb = [[]]*(L+1)
        delh = [[]]*(L+1)
        
        # Calculate the derivative of the loss function
        if(self.config['loss'] == 'cross_entropy'):
            delA[L] = -(y - H[L])
        else:
            delA[L] = -(y - H[L])*(H[L])*(1 - H[L])

        # Backpropagation loop
        for k in range(L,0,-1):
            # Compute gradients for weights, biases, and hidden layer
            delW[k] = np.matmul(delA[k],H[k-1].T) 
            delb[k] = np.sum(delA[k],axis = 1)
            delh[k-1] = W[k].T @ delA[k]
            if k>1 :
                diff_vect = np.array(A[k-1])
                d = Differential()
                if self.activation == 'sigmoid':
                    diff_vect = d.sig_dif(A[k-1])
                elif self.activation == 'tanh':
                    diff_vect = d.tan_dif(A[k-1])
                elif self.activation == 'ReLU':
                    diff_vect = d.Rel_dif(A[k-1])
                else :
                    diff_vect = d.Iden_dif(A[k-1])
                # Compute gradient for the previous layer
                delA[k-1] = np.multiply(delh[k-1],diff_vect)

        return delW,delb

    def forward_propagation(self,W,b,layers,inpl):
        A = [[]]        # Pre Activations List
        H = [inpl]      # Activations List
        
        activ = Activations()
        
        #forward_propagation loop
        for i in range (1,layers):
            #reshaping the numpy array to make it addition compatible
            rep = (H[i-1].shape[1],1)
            bias = np.tile(b[i],rep).transpose()
            
            # a = b + Wh
            a = np.add(bias,np.dot(W[i],H[i-1]))

            A.append(a)
            h = a   # this is the case handling the identity function

            if self.activation == 'sigmoid':   # Sigmoid activation function
                h = activ.sigmoid(a)
            elif self.activation == 'tanh':   # tanh activation function
                h = np.tanh(a)
            elif self.activation == 'ReLU' :            # ReLU activation function
                h = activ.g3(a)

            H.append(h)
        
        #reshaping the numpy array to make it addition compatible
        bias_new = b[layers]
        bias_new = np.tile(bias_new[:, np.newaxis], (1, self.batch))
        a = np.add(bias_new,np.inner(W[layers],H[layers-1].T))

        #compute probabilities using softmax
        A.append(a)
        a_trans = a.T
        y_hat = []
        for i in range(len(a_trans)):
            y_hat.append(activ.SoftMax(a_trans[i]))
        y_hat = np.array(y_hat)
        y_hat = y_hat.T
        H.append(y_hat)
        return A,H
    
    '''
        Function to calculate total validation loss and accuracy
    '''
    def calc_val_loss(self,W,b):
        s = 0.0
        c = 0
        for j in range(0,len(x_val)//self.batch):
            # making the input go in batches of size self.batch    
            h0 = x_val[j*self.batch : (j+1)*self.batch]
            #calling the forward_propagation function    
            A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
            y = y_val[j*self.batch : (j+1)*self.batch]
                
            yp = H[self.layers].T
            #calculating the loss and correct predictions by the model in a batch
            for itr in range(self.batch):
                if self.config['loss'] == 'cross_entropy' :
                    s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                else :
                    s = s + np.sum((y[itr]-yp[itr])**2)
                if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                    c = c + 1
        return [s,c]
            
    def Stocastic_Gradient_descent(self) :
        epochs = self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's
        
        I = Initializer()
        # Initialize W and b as per user's specification
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
        else:
            W,b = I.XavierIntializer(self.layers-1,self.npl)
        
        training_loss = []          # list consisting of all the training losses
        validation_loss = []        # list consisting of all the validation losses
        training_accuracy = []      # list consisting of all the training accuracies
        validation_accuracy = []    # list consisting of all the validation accuracies
        for i in range(epochs):
            s = 0.0
            c = 0
            for j in range(0,len(x_train)//self.batch):
                # making the input go in batches of size self.batch
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                
                #calling the forward_propagation function
                A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                #calculating the loss and correct predictions by the model in a batch
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1

                delW,delb = self.backward_propagation(A,H,W,b,y.T)
                PMA = Arithmetic()    # P - Plus, M - Minus , A - Arithmetic
                #Update rules
                W = PMA.Subtract(W,delW,self.eta)
                W = PMA.Subtract(W,W,self.eta*self.config['regularization'])
                b = PMA.Subtract(b,delb,self.eta)
            
            #calculating the total loss and total accuracies by the model at the end of the epoch
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
    
    
    def Momentum_Gradient_descent(self) :
        epochs = self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's

        I = Initializer()
        # Initialize W and b as per user's specification
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
        else:
            W,b = W,b = I.XavierIntializer(self.layers-1,self.npl)

        prev_uW,prev_ub = I.Initialize(self.layers-1,self.npl)
        beta = self.config['beta']
        
        training_loss = []                      # list consisting of all the training losses
        validation_loss = []                    # list consisting of all the validation losses
        training_accuracy = []                  # list consisting of all the training accuracies
        validation_accuracy = []                # list consisting of all the validation accuracies

        for i in range(epochs):
            s = 0.0
            c = 0
            for j in range(0,len(x_train)//self.batch):
                # making the input go in batches of size self.batch
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                #calling the forward_propagation function
                A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                #calculating the loss and correct predictions by the model in a batch
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1
            
                delW,delb = self.backward_propagation(A,H,W,b,y.T)
                PMA = Arithmetic()
                # Update rules for momentum
                '''
                    uW(t+1) = beta*uW(t) + del(W)
                    ub(t+1) = beta*ub(t) + del(b)
                    W(t+1) = W(t) - eta*uW(t+1)
                    b(t+1) = b(t) - eta*ub(t+1)
                '''

                for k in range(1,len(prev_uW)):
                    prev_uW[k] = np.add(beta*prev_uW[k],delW[k])
                    prev_ub[k] = np.add(beta*prev_ub[k],delb[k])
                
                W = PMA.Subtract(W,prev_uW,self.eta)
                W = PMA.Subtract(W,W,self.eta*self.config['regularization'])
                b = PMA.Subtract(b,prev_ub,self.eta)
                
            #calculating the total loss and total accuracies by the model at the end of the epoch        
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

    def NAG_descent(self) :    # function(layers ,neurons per layer)
        epochs =  self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's

        I = Initializer()
        # Initialize W and b as per user's specification
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
            print(len(W),len(b))
        else:
            W,b = W,b = I.XavierIntializer(self.layers-1,self.npl)
        
        prev_vW,prev_vb = I.Initialize(self.layers-1,self.npl)
        beta = self.config['beta']
        
        training_loss = []                      # list consisting of all the training losses
        validation_loss = []                    # list consisting of all the validation losses
        training_accuracy = []                  # list consisting of all the training accuracies
        validation_accuracy = []                # list consisting of all the validation accuracies
        
        for i in range(epochs):
            vW,vb = I.Initialize(self.layers-1,self.npl)
            #computing the temporary W on which derivative will be calculated
            for k in range(1,len(prev_vW)):
                vW[k] = beta*prev_vW[k]
                vb[k] = beta*prev_vb[k]
            s = 0.0
            c = 0
            
            ASA = Arithmetic()
            tempW = ASA.Subtract(W,vW,beta)
            tempb = ASA.Subtract(b,vb,beta)
            
            for j in range(0,len(x_train)//self.batch):
                # making the input go in batches of size self.batch
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                #calling the forward_propagation function
                A,H = self.forward_propagation(tempW,tempb,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                #calculating the loss and correct predictions by the model in a batch
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1
            
                delW,delb = self.backward_propagation(A,H,tempW,tempb,y.T)
                # Update rules for nesterov accelerated gradient descent
                '''
                    uW(t+1) = beta*uW(t) + eta*del(W-beta*uW(t))
                    ub(t+1) = beta*ub(t) + eta*del(b-beta*ub(t))
                    W(t+1) = W(t) - uW(t+1)
                    b(t+1) = b(t) - ub(t+1)
                '''
                for k in range(1,len(prev_vW)):
                    prev_vW[k] = beta*prev_vW[k] + self.eta*delW[k]
                    prev_vb[k] = beta*prev_vb[k] + self.eta*delb[k]

                W = ASA.Subtract(W,vW,1)
                W = ASA.Subtract(W,W,self.eta*self.config['regularization'])
                b = ASA.Subtract(b,vb,1)
                prev_vb = vb
                prev_vW = vW

            #calculating the total loss and total accuracies by the model at the end of the epoch    
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

    def RMSprop(self) :
        epochs =  self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's

        I = Initializer()
        # Initialize W and b as per user's specification
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
        else:
            W,b = W,b = I.XavierIntializer(self.layers-1,self.npl)

        vW,vb = I.Initialize(self.layers-1,self.npl)
        beta = self.config['beta']
        eps = self.config['epsilon']
        
        training_loss = []                      # list consisting of all the training losses
        validation_loss = []                    # list consisting of all the validation losses
        training_accuracy = []                  # list consisting of all the training accuracies
        validation_accuracy = []                # list consisting of all the validation accuracies
        
        for i in range(epochs):
            s = 0.0
            c = 0
            for j in range(0,len(x_train)//self.batch):
                # making the input go in batches of size self.batch
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                #calling the forward_propagation function
                A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                #calculating the loss and correct predictions by the model in a batch
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1
                
                delW,delb = self.backward_propagation(A,H,W,b,y.T)
                PMA = Arithmetic()
                # Update rules for rms prop
                for k in range(1,len(vW)):
                    vW[k] = (beta * vW[k])+ ((1-beta)*(delW[k]**2))
                    vb[k] = (beta * vb[k])+ ((1-beta)*(delb[k]**2))

                W = PMA.RMSpropSubtract(W,delW,vW,eps,self.eta)
                W = PMA.Subtract(W,W,self.eta*self.config['regularization'])
                b = PMA.RMSpropSubtract(b,delb,vb,eps,self.eta)
                delW,delb = I.Initialize(self.layers-1,self.npl)
            
            #calculating the total loss and total accuracies by the model at the end of the epoch
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

    def Adam(self) :
        epochs =  self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's

        I = Initializer()
        # Initialize W and b as per user's specification
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
        else:
            W,b = W,b = I.XavierIntializer(self.layers-1,self.npl)

        training_loss = []                      # list consisting of all the training losses
        validation_loss = []                    # list consisting of all the validation losses
        training_accuracy = []                  # list consisting of all the training accuracies
        validation_accuracy = []                # list consisting of all the validation accuracies
        
        vW,vb = I.Initialize(self.layers-1,self.npl)
        mW,mb = I.Initialize(self.layers-1,self.npl)
        beta1,beta2 = self.config['beta1'],self.config['beta2']

        for i in range(epochs):
            eps = self.config['epsilon']
            s = 0.0
            c = 0
            for j in range(0,len(x_train)//self.batch):
                # making the input go in batches of size self.batch 
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                #calling the forward_propagation function
                A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                #calculating the loss and correct predictions by the model in a batch
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1
            
                delW,delb = self.backward_propagation(A,H,W,b,y.T)
                PMA = Arithmetic()
                # Update rules for Adam
                vW_hat,vb_hat = I.Initialize(self.layers-1,self.npl)
                mW_hat,mb_hat = I.Initialize(self.layers-1,self.npl)

                for k in range(1,len(vW)):
                    mW[k] = (beta1 * mW[k]) + ((1-beta1)*(delW[k]))

                    mW_hat[k] = mW[k]/(1-np.power(beta1,i+1))
                    
                    mb[k] = (beta1 * mb[k]) + ((1-beta1)*(delb[k]))

                    mb_hat[k] = mb[k]/(1-np.power(beta1,i+1))

                    vW[k] = (beta2 * vW[k])+ ((1-beta2)*(delW[k]**2))

                    vW_hat[k] = vW[k]/(1-np.power(beta2,i+1))

                    vb[k] = (beta2 * vb[k])+ ((1-beta2)*(delb[k]**2))

                    vb_hat[k] = vb[k]/(1-np.power(beta2,i+1))
                    
                W = PMA.AdamSubtract(W,mW_hat,vW_hat,eps,self.eta)
                W = PMA.Subtract(W,W,self.eta*self.config['regularization'])
                b = PMA.AdamSubtract(b,mb_hat,vb_hat,eps,self.eta)
            
            #calculating the total loss and total accuracies by the model at the end of the epoch
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

    def NAdam(self) :
        epochs =  self.config['epochs']
        W = []    # list consisting of all the W's
        b = []    # list consiting of all the b's


        I = Initializer()
        if(self.init == "random"):
            W,b = I.Initialize2(self.layers-1,self.npl)
        else:
            W,b = W,b = I.XavierIntializer(self.layers-1,self.npl)

        vW,vb = I.Initialize(self.layers-1,self.npl)
        mW,mb = I.Initialize(self.layers-1,self.npl)
        beta1,beta2 = self.config['beta1'],self.config['beta2']
        
        training_loss = []
        validation_loss = []
        training_accuracy = []
        validation_accuracy = []
        
        for i in range(epochs):
            eps = self.config['epsilon']
            s = 0.0
            c = 0
            for j in range(0,len(x_train)//self.batch):
                # making the input go in batches of size self.batch 
                h0 = x_train[j*self.batch : (j+1)*self.batch]
                #calling the forward_propagation function
                A,H = self.forward_propagation(W,b,self.layers,h0.T)
                
                y = y_train[j*self.batch : (j+1)*self.batch]
                
                yp = H[self.layers].T
                #calculating the loss and correct predictions by the model in a batch
                for itr in range(self.batch):
                    if self.config['loss'] == 'cross_entropy' :
                        s = s - math.log(y[itr] @ yp[itr] + 1e-20)
                    else :
                        s = s + np.sum((y[itr]-yp[itr])**2)
                    if np.argmax(y[itr]) == np.argmax(yp[itr]) :
                        c = c + 1
            
                delW,delb = self.backward_propagation(A,H,W,b,y.T)
                PMA = Arithmetic()
                # Update rules for NAdam
                vW_hat,vb_hat = I.Initialize(self.layers-1,self.npl)
                mW_hat,mb_hat = I.Initialize(self.layers-1,self.npl)
                uW_hat,ub_hat = I.Initialize(self.layers-1,self.npl)
                for k in range(1,len(vW)):
                    mW[k] = (beta1 * mW[k]) + ((1-beta1)*(delW[k]))

                    mW_hat[k] = mW[k]/(1-np.power(beta1,i+1))

                    mb[k] = (beta1 * mb[k]) + ((1-beta1)*(delb[k]))

                    mb_hat[k] = mb[k]/(1-np.power(beta1,i+1))

                    vW[k] = (beta2 * vW[k])+ ((1-beta2)*(delW[k]**2))

                    vW_hat[k] = vW[k]/(1-np.power(beta2,i+1))

                    vb[k] = (beta2 * vb[k])+ ((1-beta2)*(delb[k]**2))

                    vb_hat[k] = vb[k]/(1-np.power(beta2,i+1))

                    uW_hat[k] = (beta1*mW_hat[k]) + (((1-beta1)/(1-(beta1)**(i+1)))*delW[k])

                    ub_hat[k] = (beta1*mb_hat[k]) + (((1-beta1)/(1-(beta1)**(i+1)))*delb[k])

                W = PMA.AdamSubtract(W,uW_hat,vW_hat,eps,self.eta)
                W = PMA.Subtract(W,W,self.eta*self.config['regularization'])
                b = PMA.AdamSubtract(b,ub_hat,vb_hat,eps,self.eta)
    
            #calculating the total loss and total accuracies by the model at the end of the epoch
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
    
    def Run_Models(self):
        # Generate a unique run name based on the configuration parameters
        run_name = "op_{}_ep_{}_lay_{}_npl_{}_eta_{}_bs_{}_ini_{}_reg_{}_loss_{}_activ_{}".format(self.config['optimizer'],self.config['epochs'],self.config['layers'],self.config['neurons_per_layer'],self.config['learning_rate'],self.config['batch_size'],self.config['Initialization'],self.config['regularization'],self.config['loss'],self.config['activation'])
        # Set the run name for Weights & Biases tracking
        wandb.run.name = run_name
        
        # Initialize lists to store weights and biases
        W,b = [],[]     #dummies anyways, these will be updated

        # Choose the optimizer based on the configuration
        if self.config['optimizer'] == 'sgd' :
            W,b = self.Stocastic_Gradient_descent()
        elif self.config['optimizer'] == 'momentum' :
            W,b = self.Momentum_Gradient_descent()
        elif self.config['optimizer'] == 'nag' :
            W,b = self.NAG_descent()
        elif self.config['optimizer'] == 'rmsprop' :
            W,b = self.RMSprop()
        elif self.config['optimizer'] == 'adam' :
            W,b = self.Adam()
        else:
            W,b = self.NAdam()
        return W,b


class Prediction:
    '''
        This class contains the prediction class which produces the confusion matrix
        This is called only if user specifies it
        by default this won't be called
    '''
    def predict_test(self,W,b,config):
        predicted_labels = []
        actual_labels = []
        model = Gradient_descent(784,10,config,1)

        for j in range(0,len(x_test)//config['batch_size']): 
            h0 = x_test[j*config['batch_size'] : (j+1)*config['batch_size']]
                        
            A,H = model.forward_propagation(W,b,config['layers'],h0.T)
                        
            y = y_test[j*config['batch_size'] : (j+1)*config['batch_size']]
                        
            yp = H[config['layers']].T
            
            for itr in range(config['batch_size']):
                predicted_labels.append(yp[itr])
                actual_labels.append(y[itr])

        predicted_labels = np.array(predicted_labels)
        actual_labels = np.array(actual_labels)
                        
        conf_matrix = confusion_matrix(np.argmax(predicted_labels, axis=1), np.argmax(actual_labels, axis=1))
        acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
        print(conf_matrix)
        print(acc*100)
        return actual_labels,predicted_labels
