'''
Created on 2021年10月3日

@author: Konic
'''
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialization(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    grads = {'dw':dw,'db':db}
    return grads,cost

def optimize(w,b,X,Y,learning_rate,epochs,print_cost=False):
    costs = []
    for epoch in range(epochs):
        grads,cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        w -= learning_rate * dw
        b -= learning_rate * db
        if epochs % 100 == 0:
            costs.append(cost)
            if print_cost:
                print('epochs:%i;cost:%f'%(epoch,cost))
    params = {'w':w,'b':b}
    return params,costs

def predict(w,b,X):
    predictions = sigmoid(np.dot(w.T,X)+b)
    return (predictions>0.5).astype(int)

def model(X_train,Y_train,X_test,Y_test,epochs=200,learning_rate=0.01,print_cost=False):
    dim = X_train.shape[0]
    w,b = initialization(dim)
    params, costs = optimize(w,b,X_train,Y_train,learning_rate,epochs,print_cost)
    w,b = params['w'],params['b']
    Y_predictions = predict(w,b,X_test)
    print('Test Acc:{}%'.format(100-np.mean(abs(Y_predictions-Y_test))*100))

if __name__ == '__main__':
    n = 20 #特征维度
    m = 200 #样本数目
    X_train = np.random.random((n,m))
    Y_train = np.random.randint(0,2,size=(1,m))
    X_test = np.random.random((n,10))
    Y_test = np.random.randint(0,2,size=(1,10))

    model(X_train,Y_train,X_test,Y_test,epochs=200,learning_rate=0.01,print_cost=False)