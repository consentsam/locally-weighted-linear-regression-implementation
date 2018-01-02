#Here in this problem we just want to learn/plot the Hypothesis function 
# for the given dataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time 
import matplotlib.pyplot as plt
import pickle
# I have put the start_time as a variable to mark down starting time
start_time =time.time()
#The value of eta (learning rate) is tuned to get the appropriate 
#model  
eta=0.01
#Reading x_data as input which is actually the area of the houses
X=np.genfromtxt('q1x.dat')
# Now we are counting number of rows in dataset X
size_x=X.size
# Now we are intended to calculate mean and variance of the data as to 
# Normalise the X dataset
mean_x=np.mean(X)
var_x=np.var(X)
#Normalised Data
for i in range(size_x):
	X[i]=(X[i]-mean_x)/var_x

# Normalising the Y dataset
Y=np.genfromtxt('q1y.dat')
# Now we are setting a stopping parameter so that when the
# difference between the error values decreases to
# value less than stopping parameter then the iteration will stop.
# We do so because in the start the functions start to decrease
# fastly but after some time the functions start to decrease very
# slowly and so the difference between the error functions seems to 
# decrease very slowly and so if we keep a limit that if the 
# difference between them becomes less than stopping then we 
# will assume this as converged
stopping=0.00032
# Initialising the parameter to be a zeros of dimension 2 X 1
#  
param=np.zeros((2,1))
# Initialising param_old to be array of 1
# Just to ensure that the np.linalg.norm(param_old-param)>stopping
param_old=np.ones((2,1))
# We are keeping the batchsize of 10
batchsize=10
count=0
number_of_iterations=0

while(np.linalg.norm(param_old-param)>stopping):
	number_of_iterations=number_of_iterations+1
	gradient=np.zeros((2,1))
	for i in range(batchsize):
		count=count%size_x
		xi=np.array([[1],[X[count]]])
		gradient=gradient+(Y[count]-param.transpose().dot(xi))*xi
		count=count+1
	gradient=gradient/batchsize
	param_old=param
	param=param+eta*gradient

print("Number of Iterations "+str(number_of_iterations))
print(param)

# Extracting the theta values and then appending mean_x,var_x
# in a newlist 
newlist = []
for i in param:
	for j in i:
		newlist.append(j)
newlist.append(mean_x)
newlist.append(var_x)

# 'ro' argument plots the points 
plt.plot(X,Y,'ro')
# Newlist2 is a matrix that append 1 in the starting of the matrix 
# because we need it to be multiply with the test data and predict
# for that,Here one column is 1 which is going to be multiplied with theta_0
# and other ,test data X is going to be multiplied with theta_1 for prediction
# newlist_2 is a matrix of dimension m X 2 as param is a matrix
# of dimension 2 X 1 its transpose will be 
newlist2 = []
for i in range(size_x):
	newlist2.append([1,X[i]])
newlist2 = np.array(newlist2)
# Here result signifies the predicted row matrix 
result = (param.transpose().dot(newlist2.transpose())).transpose()
# It can also be written as newlist2.param as there is a formula 
# (A_t.B_t)^t=(B_t_t.A_t_t)=(B.A) as we just want to multiply to get
# the product in the form of theta_0*x_0+theta_1*x_1

# Passing X,Y without any argument makes the lines joining all the points

plt.plot(X,result)

# We are defining an Error Function which takes inputs theta_0 and theta_1
# which gives the respetive error which we will use to plot 3D Axes

def Error_Function(theta0,theta1):
	error = 0
	param = np.array([[theta0],[theta1]])
	for i in range(size_x):
		xx = np.array([[1],[X[i]]])
		error = error + (Y[i]-param.transpose().dot(xx))**2
	error = error/(2*size_x)
	return error

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# We are keeping the range of 0 to 10 with 0.2 difference 
# because we know theta_1 is around 5 and so it will fall in between
# Similarly 
# We are keeping the range of 0 to 30 with 0.2 difference because
# we know theta_1 is around 17 and so it will fall in between 
theta0 = np.arange(0.0,10.0,0.2)
theta1 = np.arange(0.0,30.0,0.2)
# Function meshgrid makes the combination of every possible theta_0 and
# theta_1 like axes x and axes y 
Theta0,Theta1 = np.meshgrid(theta0,theta1)
# print(Theta0.shape)
zs = np.array([Error_Function(theta0,theta1) for theta0,theta1 in zip(np.ravel(Theta0),np.ravel(Theta1))])
Z = zs.reshape(Theta0.shape)

ax.plot_surface(Theta0,Theta1,Z)
ax.set_xlabel('Parameter Theta0')
ax.set_ylabel('Parameter Theta1')
ax.set_zlabel('Error Function')

# We will be dumping the parameter and mean_x and variance of x according 
# to that we have normalised the train data ,This mean and varainces will be
# respectively used nomalise the data 
with open('outputfile.txt','wb') as fp:
	pickle.dump(newlist,fp)

end_time = time.time()
print("Time Elapsed "+ str(end_time-start_time))
plt.show()

