import pandas
from  sklearn.linear_model import LogisticRegression


#open and read file
df = pandas.read_csv('RSL_copy.csv')
print(df.dtypes)
data= df.values   #df['outcome'].values

#i had problem using raw data.a wawrning it did not converge, i needed to scale it
# from sklearn.preprocessing import MinMaxScaler
# Data_scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = Data_scaler.fit_transform(data)
# print (scaled_data[0:10])  # first 10 row of scaled Data

#OR USE
# from sklearn.preprocessing import StandardScaler
# data_scaler = StandardScaler().fit(data)
# scaled_data = data_scaler.transform(data)
# print ("\nRescaled data:\n", scaled_data [0:5])



#split into features and target
X_array = data[:,0:2]
#print(X_array)  #is in  2D array shape
print(X_array.shape)
Y_array = data[:,2]  # y -target these are label it is not god to rescaled it
#print(Y_array)  #will be in list shpae
print(Y_array.shape)
#split data into training set and test set
#As you can notice size of the A and B, C   are not in same range(RSL are about 10 times the outcome)
#normalisation must be select to normalise the values
from sklearn.model_selection import train_test_split
#help(train_test_split)
X_train, X_test, y_train, y_test = train_test_split(X_array,Y_array,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#create an insance of the model
lrmodel=LogisticRegression(solver='newton-cg')  #default nomalisation is  l2
# Train the MOdel to get line of best FIT
lrmodel.fit(X_train,y_train)
# make your prediction with x_train and compare it with y_train
train_prediction = lrmodel.predict(X_train)
#find the accuracy of the model by  comparing  it with y_train
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(train_prediction,y_train)
print('train prediction is',accuracy*100,'%')

#after training the model, test the model
prediction =lrmodel.predict(X_test)
#find the accuracy of your prediction
accuracy = accuracy_score(prediction,y_test)
print('test predcition:', accuracy*100,'%')
from sklearn.metrics import confusion_matrix
#confusion_matrix  #cannot handle multiclass probelms
confusion_matrix(y_test,prediction)
print('matrix',confusion_matrix(y_test,prediction))
#train prediction is 65.83333333333333 % using scaled data with lbfgs
# test predcition: 64.28571428571429 %

#using scaled data with solver: newton-cg with unscaled data
#lrmodel=LogisticRegression(solver='newton-cg')
# train prediction is 77.73809523809524 %
# test predcition: 75.71428571428571 %


#using a  StandardScaler,instead of MinMax scaler eith solver:lbfg
# (produce better results with rescaled data for Llike linear regression, logistic regression that assumes a Gaussian distribution in input dataset

# train prediction is 73.80952380952381 %
# test predcition: 71.9047619047619 %

#plot each data
from matplotlib import pyplot as plt
#plot col0 =rslA against classes 1-7 of outcome
#plot col0 =rslb against classes 1-7 of outcome
plt.scatter(X_array[Y_array == 0][:, 0], X_array[Y_array == 0][:, 1], color='g', label='0')
plt.scatter(X_array[Y_array == 1][:, 0], X_array[Y_array == 1][:, 1], color='y', label='1')
plt.scatter(X_array[Y_array == 2][:, 0], X_array[Y_array == 2][:, 1], color='r', label='2')
plt.scatter(X_array[Y_array == 3][:, 0], X_array[Y_array == 3][:, 1], color='b', label='3')
plt.scatter(X_array[Y_array == 4][:, 0], X_array[Y_array == 4][:, 1], color='k', label='4')
plt.scatter(X_array[Y_array == 5][:, 0], X_array[Y_array == 5][:, 1], color='c', label='5')
plt.scatter(X_array[Y_array == 6][:, 0], X_array[Y_array == 6][:, 1], color='m', label='6')
plt.scatter(X_array[Y_array == 7][:, 0], X_array[Y_array == 7][:, 1], color='w', label='7')
plt.xlabel('RSL B hub site')
plt.ylabel ('RSL A local site')
plt.legend()
plt.grid()
plt.show()

plt.hist(X_array)   #10 BARS
plt.title('RSL distribution of site A AND b')
plt.xlabel('no of sample')
plt.ylabel('RSL A local site')
plt.grid()
plt.show()

plt.hist(Y_array)
plt.title('outcome distribution')
plt.xlabel('no of sample')
plt.ylabel('outcomes')
plt.show()
df['terminal _A_ site_ RSl'].hist()
plt.title('A_ site_ RSl distribution')
plt.xlabel('no of sample')
plt.ylabel('A_ site_ RSl')
plt.show()
df['Hub_B_site_RSL'].hist()
plt.title('_B site_ RSl distribution')
plt.xlabel('no of sample')
plt.ylabel('B_ site_ RSl')
plt.show()

plt.hist(X_array[:,0],10,color='y', label='RSL_A')
plt.hist(X_array[:,1],10,color='g', label='RSL_B')
plt.title('RSL distribution of site A AND b')
plt.xlabel('RSL A and B ')
plt.ylabel('no of sample ')
plt.legend()
plt.grid()
plt.show()

#evaluation of the model: give it arbitary figure
input_data=  [-23,-31]  #rsl_a,rsl_b
#convert list ot array
import numpy
Input_array = numpy.array(input_data)
print(Input_array.shape)  #(2,) (rows=2,cols=0)
#X_array = data[:,0:2]
#print(X_array)  #is in  2D array shape
print(X_array.shape)  #(1050, 2) (rows=1050,cols=2)
#so we need to reshape Input_array to be like X_array (1row,2cols)
#convert 2 rows (2,) (rows=2,cols=0)  into 2 cols (rows=1,cols=2) instead
Input_array_reshaped = Input_array.reshape(1,-1)
print(Input_array_reshaped.shape) #(1, 2)
make_prediction = lrmodel.predict(Input_array_reshaped)
print(make_prediction)  #make_prediction= [10],  pos is 0
print(make_prediction.dtype) #array
if make_prediction == 0:
    print('site A is up')
elif make_prediction == 1:
    print('site A is down: faulttype: 1. inteference 2. misalignment 3. one of the odu is faulty')
elif make_prediction == 2:
    print('site A is down: faulttype: 2. NO power at remote site(A),2.ODU offline remote site(A)(check alarm \'IF cable open\') ')
elif make_prediction == 3:
    print('site A is down: faulttype: 3.ODu hunged at remote site(A), reset power at b ODU/IDU/If cable offline,at remote endoth sites(A,B)')
elif make_prediction == 4:
    print('site A is down: faulttype: 4. cascaded cable  faulty at hub  Site (B), 2.')
elif make_prediction == 5:
    print('site A is down: faulttype: 5 ODU at hub site(B)degraded( reset ODU, reterminate IF cable,check alarm)')
elif make_prediction == 6:
    print('site A is down: faulttype: 6 if power is okay, odu burnt at either remote site (A) OR (B)')
else:  #do feature elimination for data irrelevant to outcome
    print('case 7: site A cannot be determined by RSL data')


#sve your model
import pickle
filename = 'trained_lrmodel.sav'
pickle.dump(lrmodel,open(filename, 'wb'))

#load the saved model
loaded_model = pickle.load(open('trained_lrmodel.sav','rb'))


#NEXT STEP ON TIHS trained_lrmodel.sav on microwaveApp.py
#configure to webrowser using STreamlit
#to run the application on web browser ...
# #run  on your pycharm terminal.... ' streamlit run microwaveAPP.py '