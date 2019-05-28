import sklearn as sl
import numpy as np
from sklearn.neural_network import MLPClassifier
import sys
import random
from sklearn.preprocessing import StandardScaler

class txtParser:
    def __init__(self, set):
        #Constructor
        self.setList = []
        self.labelList = []

        self.populateList(set,self.setList,self.labelList)

        self.setList = np.array(self.setList).astype(np.float64).reshape(-1,4)
        self.labelList = np.array(self.labelList)
    def populateList(self, file, list, labels):
        with open(file) as fp:
            for line in fp:
                if len(line) < 5:
                    continue
                features, label = self.lineParser(line)
                list.append(features)
                labels.append(label)
        return
    #Parse line by splitting spaces
    def lineParser(self, line):
        if "," in line:
            line = line.split(",")
            line[-1] = line[-1].strip("\n")
        else:
            line = line.split()
        return line[0:-1],line[-1]
def objective(x0,X,Y,seed):
    lri,hls,tol,vf,m = x0[0],x0[1],x0[2],x0[3],x0[4]
    clf = MLPClassifier(solver='sgd',
                        learning_rate_init=lri,
                        max_iter = len(X.setList)*100,
                        hidden_layer_sizes=(hls),
                        activation = 'logistic',
                        tol = tol,
                        random_state=seed,
                        early_stopping = True,
                        validation_fraction = vf,
                        momentum = m)
    clf.fit(X.setList,X.labelList)
    print("Iterations for convergence: %s"%clf.n_iter_)
    print("Seed: %d"%seed)
    print("Number of layers: %s"%clf.n_layers_)
    return (clf.score(Y.setList,Y.labelList)*100)
def runTest(LR,HN,TolE,VSP,Momen,X,Y,iter):
    accT = np.zeros(25)
    for i in range(len(LR)):
        for j in range(5):
            for it in range(iter):
                acc = objective([LR[i],HN[j],TolE[j],VSP[j],Momen[j]],X,Y,12345)
                print("CASE %d: %d: %d"%(i,j,it))
                accT[(i*5)+(j)]+=acc
            accT[(i*5)+(j)] /= (iter)
    print("Overall Accuracy: %s"%str(accT))
    return

def main():

    #LR = [0.05,0.1,0.15,0.2,0.25]
    #HN = [1,75,150,300,450]
    #TolE = [1e-3,1e-4,1e-5,1e-6,1e-7]
    #VSP = [0.1,0.15,0.2,0.25,0.3]
    #Momen = [0.75,0.8,0.85,0.9,1]

    X = txtParser("iris-training.txt")
    Y = txtParser("iris-test.txt")
    #runTest(LR,HN,TolE,VSP,Momen,X,Y,10)
    maxParams = [0.25,75,1e-7,.3,1]
    print("X: iris-training.txt")
    print("Y: iris-test.txt")
    print("LR:%s Hidden Nodes:%s Tol:%s Validation Proportion:%s Momentum:%s "%(maxParams[0],maxParams[1],maxParams[2],maxParams[3],maxParams[4]))
    print("| ########################################## |")
    print("| ########################## 10 RANDOM INDEPENDANT EXPERIMENTS ########################## |")
    accT = 0.0
    for i in range(10):
        print("Run %d"%(i+1))
        seed = random.uniform(0,10000)
        acc = objective(maxParams,X,Y,int(seed))
        accT += acc
        print("%2f %% Accuracy (UNSEEN DATA ACCURACY - X training Y test)"%(acc))
        print("| ########################################## |")
    accT /= 10
    print("Overall Accuracy: %2f"%accT)
    return

if __name__ == "__main__":
    main()
