# Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

import seaborn as sns


from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data

dataset_iris = np.array(data)
print(dataset_iris)

legends = iris.target_names
legendslist_iris = np.array(legends)
print(legendslist_iris[1])

# Target vectors
t1 = np.array([1, 0, 0])
t2 = np.array([0, 1, 0])
t3 = np.array([0, 0, 1])

legends = ['Setosa', 'Versicolour', 'Virginica']
# Split data into training and testing data
#N_obsv = length(class1)
N_train = 30
N_test  = 20

Classes = 3

#def load_data():
samples = []
for i in range(Classes): #3 classes
    file_name = './class_' + str(i+1)
    print(file_name)

    with open(file_name, 'rb') as data_file:
        for row in data_file:
            elements = [ element.strip() for element in row.decode().split(',') ] #row.decode()?
            sample = np.array(elements[ :len(elements)], dtype=np.float32)
            samples.append(sample)
            print(sample)

print("\n")
print(samples[0])
#   return samples



for i in range(Classes): #3 classes
    file_name2 = './class_' + str(i+1)
    tmp = np.loadtxt(file_name2,delimiter=",")
    
   # print("\n")
   # print(tmp)
    # Add the class, and 1
    class_number = np.ones((tmp.shape[0],2)) 
    print(class_number)
    print("\n----------------------------------------------------------\n")
    class_number[:,-1] *= i 
    tmp = np.hstack((tmp, class_number))
    print("\n")
    print(tmp)
    if i > 0:
        data = np.vstack((data, tmp))
    else:
        data = copy.deepcopy(tmp)
print("\n")
print(data)

"""
# Normalize
tmp = data[:,:-1] 
#tmp = tmp - tmp.mean(axis=0)
tmp = tmp / tmp.max(axis=0)
data[:,:-1] = tmp
"""

print("----------------------------------------------------------\n")
print(data)

train_set_size = 30

N = int(len(data)/Classes)
print(N) #50
print(int(len(data[0])))
print(data.shape[1])
test_set_size = N - train_set_size
train_samples = np.zeros((Classes*train_set_size,int(len(data[0]))))
test_samples = np.zeros((Classes*test_set_size,int(len(data[0]))))
print("\n")
print(train_samples.shape)
print("\n")
print(test_samples.shape)

for i in range(Classes):
    temp = data[(N*i):((i+1)*N)]
    print(temp.shape)
    train_samples[(train_set_size*i):((i+1)*train_set_size), :] = temp[ :train_set_size,:]
    test_samples[(test_set_size*i):((i+1)*test_set_size), :] = temp[train_set_size:,:]

print("\n")
print(train_samples)
print("\n")
print(test_samples)


for i in range(Classes):
    petal_data = data[(50*i):(50*(i+1)),2:-2]
    plt.scatter(petal_data[:,0],petal_data[:,1])
plt.title("Petal data")
plt.show()


for i in range(Classes):
    sepal_data = data[(50*i):(50*(i+1)),:-4]
    plt.scatter(sepal_data[:,0],sepal_data[:,1])
plt.title("Sepal data")
plt.show()


print(int(len(train_samples[0])))