import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time 
import matplotlib.pyplot as plt
import pickle
start_time =time.time()
eta=0.3
X=np.genfromtxt('q3x.dat')
size_x=X.size
mean_x=np.mean(X)
var_x=np.var(X)
tou=0.05
#Normalised Data
for i in range(size_x):
	X[i]=(X[i]-mean_x)/var_x
m=X.shape[0]
Y=np.genfromtxt('q3y.dat')
stopping=0.00032


batchsize=m
count=0
number_of_iterations=0
#Since x is varying from x =-0.4 to 0.4 so we will be doing  partitions with intervals 0.02
data_min=np.min(X)
data_max=np.max(X)
interval = 0.02
query_x=np.arange(data_min,data_max,interval)
#print(query_x)
query_y=[]
for k in query_x:
	param=np.zeros((2,1))
	param_old=np.ones((2,1))
	while(np.linalg.norm(param_old-param)>stopping):
		number_of_iterations=number_of_iterations+1
		gradient=np.zeros((2,1))
		for i in range(batchsize):
			count=count%m
			xi=np.array([[1],[X[count]]])
			gradient=gradient+np.exp(-((k-X[count])**2)/(2*tou*tou))*(Y[count]-param.transpose().dot(xi))*xi
			count=count+1
		gradient=gradient/batchsize
		param_old=param
		param=param+eta*gradient

	print("param"+str(param))
	newlist4 = []
	newlist4.append([1,k])
	newlist5=np.array(newlist4)
	resullt=(param.transpose().dot(newlist5.transpose())).transpose()
	query_y.append(resullt[0][0])
	print("Result"+str(resullt))

end_time = time.time()
print("Time Elapsed "+ str(end_time-start_time))
print("Number of Iterations "+str(number_of_iterations))
#Exporting the newlist to output file to normalize x and then inputing it into function to get the output
newlist = []
for i in param:
	for j in i:
		newlist.append(j)
newlist.append(mean_x)
newlist.append(var_x)

query_y=np.array(query_y)
plt.plot(X,Y,'ro')
plt.plot(query_x,query_y)
plt.show()