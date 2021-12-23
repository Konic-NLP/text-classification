'''
Created on 2021年10月3日

@author: Konic
'''
import math
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from cherrypy import engine
import matplotlib.pyplot as plt
import csv
import fileinput,re

lr=[0.1,0.2,0.01]
bias=[0,1,0.1,0.5]
gridsearch={'lr':lr,'bias':bias}

[(a,b) for a in lr for b in bias]

def file_read(file_add):
    '''
    preprocess the text and stiore it into a dict
    '''
    train=defaultdict(list)
    with fileinput.input(file_add,openhook=fileinput.hook_encoded(encoding='utf-8'))as f:
        for i in f:
            data=i.strip().split('\t',1)
            train[data[0]].append(data[1])
            train[data[0]].append(1)
    return train

# with fileinput.input(r'D:\CLASIC课程\CSCI5832 NLP\homework\hotelNegT-train.txt',openhook=fileinput.hook_encoded(encoding='utf-8'))as f:
#     for i in f:
#         data=i.strip().split('\t',1)
#         train[data[0]].append(data[1])
#         train[data[0]].append(0)


def sentimentaldict():
    #create two sentiment dicts for extract features
    
    pos_word=set()
    neg_word=set()
    
    with open(r'D:\CLASIC课程\CSCI5832 NLP\homework\positive-words.txt','r',encoding='utf-8') as f:
        for i in f:
            pos_word.add(i.strip())
            
    with open(r'D:\CLASIC课程\CSCI5832 NLP\homework\negative-words.txt','r',encoding='utf-8') as f:
        for i in f:
            neg_word.add(i.strip())
            
    return pos_word,neg_word
    

def feature_extract(train):       
    '''
    extract the features according to rules
    '''
    pos_word,neg_word=sentimentaldict()
    
    feature=defaultdict(list)
    pro=["I", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]
    # print(map(lower,pro))
    for k,v in train.items():
        text=[i for i in str(v[0]).lower().split()]
    #     print(k,text)
        f1=len([i for i in text if re.sub('[.,?!]','',i.strip('')) in pos_word]) #nice. / cheap 
        f2=len([i for i in text if re.sub('[.,?!]','',i.strip('')) in neg_word])
        f3=lambda x: 1 if 'no' in [re.sub('[.,?!]','',i.strip('')) for i in x] else 0
    #     f4=len([i for i in pro if i.lower() in [re.sub('[.,?!]','',k.strip()) if '\'' not in k else re.sub('[.,?!]','',k.strip().split('\'')[0])  for k in text] ])
 
        f4=(len([k for k in [i.split('\'')[0] if '\'' in i else re.sub('[.,?!]','',i.strip('')) for i in text] if k in map(lambda x:x.lower(),pro)])) 
    
        f5=lambda x: 1 if '!' in ''.join(x) else 0
        f6=round(math.log(len(text)),3)
        feature[k]=[f1,f2,f3(text),f4,f5(text),f6]
    
    return feature


          
          


def writecsv(feature,fileloc,train):
    '''

write features into one csv file and each line begin with the ID, then proceed by f1-f6 features.
'''

    with open(fileloc,'w',encoding='utf-8',newline='')as f:
    #     fieldname=['id','f1','f2','f3','f4','f5','f6']
    #     writer = csv.DictWriter(f,fieldnames=fieldname)
        headers=['ID','x1','x2','x3','x4','x5','x6','label']
        fcsv=csv.writer(f)
        fcsv.writerow(headers)
        for k,v in feature.items():
        #         fcsv.writerows([k+[v])
            v.insert(0,k)
            v.append(train[k][-1])
    #         new=[]
    #         new.append(k)
    #         for i in v:
    #             new.append(i)
                 
            fcsv.writerows([v])
    #     






def readcsv(fileloc,trainsize=0.8):
    '''
read the .csv file and split the train and test

'''
#     random.seed(0)     # compare the result and control the factor by set the random seed
    data=pd.read_csv(fileloc,engine='python')
    
    
    trainX,trainY=[],[]
    testX=defaultdict(int)
    testY=defaultdict(list)
    #  
    # # def data_split(train,y):
    #  
    num=math.ceil(len(data)*trainsize)
    
    # print(data)
    # data['bias']=1
    
    # num_pos=len(data.loc(data['label']==1))
    
    ID=list(data.index)
    # np.random.seed(0)
    np.random.shuffle(ID)
    
    for i in ID:
        
        data_=data.loc[i].values
#         print(data_)
#             '''
#             # one kind of sample, split the data squentially and one by one, regardless the proportion of two catogries in top N data(N is the size of trainset)
#             '''
        if len(trainX)>=num:
            testX[data_[0]]=data_[1:-1]
            testX[data_[0]]=np.append(testX[data_[0]],1)
            testY[data_[0]]=data_[-1]
        else:
            trainX.append(data_[1:-1])
            trainX[-1]=np.append(trainX[-1],1)
#             print(data_[-1])
            trainY.append(data_[-1])
            
        
        
    X=np.array(trainX)
    trainY=np.array(trainY)
    trainY=trainY.reshape(num,1)
#         print(trainY.shape,X.shape)
    return X,trainY,testX,testY
        
#         if balance==True:
#             
#             '''
#             another sample: split the testset and trainset according to the proportions of two catogories
#             So in the testset, the proportion of two class is the same as that in the trainset
#             
#             while in the real experiments, there is no big difference.
#             '''
#             num_pos=math.ceil((np.sum( data['label']==1))*trainsize)
#             num_neg=math.ceil((np.sum( data['label']==0))*trainsize)
#         
#             if data_[-1]==1 and len(trainX)<(num_neg+num_pos):
#                 trainX.append(data_[1:-1])
#                 trainX[-1]=np.append(trainX[-1],1)
#                 trainY.append(data_[-1])
#                   
#             elif data_[-1]==0 and len(trainX)<(num_neg+num_pos):
#                 trainX.append(data_[1:-1])
#                 trainX[-1]=np.append(trainX[-1],1)
#                 trainY.append(data_[-1])
#                 
#             else:
#                 testX[data_[0]]=data_[1:-1]
#                 testX[data_[0]]=np.append(testX[data_[0]],1)
#                 testY[data_[0]]=data_[-1]
#         if balance==False:


#             print(data_)
#             '''
#             # one kind of sample, split the data squentially and one by one, regardless the proportion of two catogries in top N data(N is the size of trainset)
#             '''



# def z(X,W):
#     return float(np.dot(X.T,W))
# print(1/(1+np.exp(-np.dot(X.T,W).astype(np.float))))


def sigmoid(z):    
    a= 1/(1+np.exp(-z))
    return a








def train(X,Y,lr=0.001,optimizer='SGD',bias=1,batch_size=20,epoch=2000):
    num_train=X.shape[0]
    lr=float(lr)
    num_feature=X.shape[1]
    # the number of input and the dimension of the feature
    # num_train=152=m, num_feature=6 
    
    W=np.zeros((num_feature,1))
    #weight=(6*1)
#     w=W[:-1]
#     b=W[-1]
    W[-1]=bias
    
    cost_record=[]
    
 
    
    
    '''
    In the multiple times experiments, there eixts no obvious difference between GD and minibatch-GD in terms of performance;
    Moreover, the training speed is slower in mini-batch, I guess the reason realted to the small training set. 
    So it's useless to apply mini-batch when coping with small data
    
    '''
    
    
    
          
    # 
   
           
    '''
    mini-batch
        
    '''
    if optimizer=='minigd':
        batch_size=batch_size
#         batch_num=int(num_train/batch_size)
           
        for i in range(epoch):
            cost_sum=0.0
            for k in range(0,num_train,batch_size):
                X_i=X[k:k+batch_size]
                y_i=Y[k:k+batch_size]
                y_hat=sigmoid(np.dot(X_i,W).astype(float))
                
                cost = -(np.sum(y_i*np.log(y_hat)+(1-y_i)*np.log(1-y_hat)))/batch_size
                cost_sum+=cost
                dW= np.dot(X_i.T,(y_hat-y_i))/batch_size
#                 db = np.sum(y_hat - y_i)/batch_size
                 
                W=W-lr*dW
#                 b=b-lr*db
                if i %100==0:
                     
                    print('epoch %d step %d cost: %f' % (i, k/batch_size, cost))
            if i%50==0:
                cost_record.append((cost_sum,i))
           
    '''
    SGD&GD  from the experiment using time.time() to calculate the runtime, GD maybe the most efficient way since the size of the training set is not very big
    
    '''
           
    if optimizer=='SGD':
        
        shuffle_ix = np.random.permutation(np.arange(len(X)))
        train_data = X[shuffle_ix,:]
        train_label = Y[shuffle_ix,:]
        for i in range(epoch):
#             cost_sum=0
            for k,v in zip(train_data,train_label):
               
#                 y_hat=sigmoid(np.dot(k.T,w).astype(float)+b)
                y_hat=sigmoid(np.dot(k.T,W).astype(float))
                
                k=k.reshape(k.shape[0],1)
                cost_sum = -(v*np.log(y_hat+1e-5)+(1-v)*np.log(1-y_hat+1e-5))
#                 print(cost)
#                 cost_sum+=cost
                
                dW= np.dot(k,(y_hat-v).reshape(1,1))
#                 db = np.sum(y_hat - v)
                W=W-lr*dW
#                 b=b-lr*db
                    
            if i %100==0:
                cost_record.append((cost_sum,i))
                print('epoch %d cost: %2f' % (i, cost_sum))
                
    if  optimizer=='GD':
#         shuffle_ix = np.random.permutation(np.arange(len(X)))
#         train_data = X[shuffle_ix,:]
#         train_label = Y[shuffle_ix,:]
#         
        for i in range(epoch):
#             the followings are GD
            y_hat=sigmoid(np.dot(X,W).astype(float))
            # shape:  y_hat(152,1)  trainY(152,1)
            cost_sum = -(np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat)))/num_train    
        #     print(cost)
            dW= np.dot(X.T,(y_hat-Y))/num_train
#             db = np.sum(y_hat -Y)/num_train   
            W=W-lr*dW
#             b=b-0.1*db
            if i %100==0:
                cost_record.append((cost_sum,i))
                print('epoch %d cost: %f' % (i, cost_sum))
    print('final cost:',cost_sum)
    return  W,cost_record
                
    


                
                
                
#             the followings are GD

            
    
def predict(testX,testY,w):
    '''
    using the weights to caculate the y-hat on dev/test set and set the threshold to get the predicting label 
    then calculate the f-score and accuracy according to the gold-label 
    '''
#     print(W)
    testID,testdata=[],[]
    for k,v in testX.items():
        testID.append(k)
        testdata.append(v)
    testanswer=[testY[i] for i in testID]
    testdata=np.array(testdata)
    num=testdata.shape[0]
    testanswer=np.array(testanswer)
    
    testanswer=testanswer.reshape(num,1)
    # for i in range(len(np.dot(testdata,W))):
    #     print(np.dot(testdata,W)[i],testanswer[i])
    z=np.dot(testdata,w).astype(float)
    test_hat=sigmoid(z)
    
    right=0
    TP,FP,TN,FN=0,0,0,0
    result=defaultdict(str)
    for i in range(len(test_hat)):
        if test_hat[i]>0.5:
            answer=1  
            result[testID[i]]='POS'  
            
        else:
            answer=0
            result[testID[i]]='NEG'
#         print(testID[i],answer,testanswer[i])
        if answer==testanswer[i]:
            right+=1
        if answer==1 and testanswer[i]==1:
            TP+=1
        if answer==1 and testanswer[i]==0:
            FP+=1
        if answer==0 and testanswer[i]==0:
            TN+=1
        if answer==0 and testanswer[i]==1:
            FN+=1
            
    r=TP/(TP+FN)
    p=TP/(TP+FP)
    F=2*r*p/(p+r)
    return right,right/num,F,result
   
        
    
def draw(cost_record):
#plt.plot(X,Y)
    cost_,epoch=zip(*cost_record)
    plt.plot(epoch,cost_)
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.show()









    
#  
#          
# #     return trainX,trainY,testX,testY
#  
# # trainX,trainY,testX,testY=data_split(feature, 0.8)
#  








if __name__ == '__main__':
#     train_neg=file_read(r'D:\CLASIC课程\CSCI5832 NLP\homework\hotelNegT-train.txt')
    test_data=file_read(r'D:\CLASIC课程\CSCI5832 NLP\homework\HW2-testset.txt')
#     train_pos=file_read(r'D:\CLASIC课程\CSCI5832 NLP\homework\hotelPosT-train.txt')
#     train1 = {**train_pos, **train_neg}
    feature=feature_extract(test_data)
    
    writecsv(feature,r'D:\CLASIC课程\CSCI5832 NLP\homework\test_data.csv',test_data)
    trainx,trainy,testx,testy=readcsv(r'D:\CLASIC课程\CSCI5832 NLP\homework\feature.csv', 1)
    w,cost_record=train(trainx, trainy, lr=0.0001, optimizer='SGD',bias=0, batch_size=20, epoch=2000)
#     right,accuracy,fscore,result=predict(testx, testy, w)
#     print('right nums: %d, accuracy:%.2f, F-score:%.2f'%(right,accuracy,fscore))
#     for k,v in result.items():
#         print(k,'\t',v)
#     draw(cost_record) 
    trainx,trainy,testx,testy=readcsv(r'D:\CLASIC课程\CSCI5832 NLP\homework\feature.csv', 0.8)
    right,accuracy,fscore,result=predict(testx, testy, w)
    print('right nums: %d, accuracy:%.2f, F-score:%.2f'%(right,accuracy,fscore))
#     for k,v in result.items():
#         print(k,'\t',v)
    trainx,trainy,testx,testy=readcsv(r'D:\CLASIC课程\CSCI5832 NLP\homework\test_data.csv', 0)
    right,accuracy,fscore,result=predict(testx, testy, w)
    print('---------------------------------')
    for k,v in result.items():
        print(k,'\t',v)
    with open(r'D:\CLASIC课程\CSCI5832 NLP\homework\Ge-Sijia-assgn2-out.txt','w',encoding='utf-8')as f:
        for k,v in result.items():
            print(k+'\t'+v,file=f)
    
    
  
    
    