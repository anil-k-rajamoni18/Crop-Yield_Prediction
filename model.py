# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('crops2.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7:9].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y[:,1] = labelencoder.fit_transform(y[:,1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
y = onehotencoder.fit_transform(y).toarray()
y=y[:,:8]
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X)

# Fitting Decision Tree Regression to the dataset
'''from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

D_pred = regressor.predict(X)'''

#crops={'rice':0,'tomato':1,'garlic':2,'wheat':3}
crops={'carrot':0,'potato':1,'beans':2,'cauliflower':3,'bottleguard':4,'tomato':5,'cowpeas':6,'chillies':7}

crops_list=['carrot','cauliflower','tomato']

arr=np.array([0,0,0,0,0,0,0,0])
for i in crops_list:
    arr[crops[i]]+=1

crop_predict=regressor.predict(np.array([arr]))

crops_r={0:'beans',1:'bottleguard',2:'carrot',3:'cauliflower',4:'chillies',5:'cowpeas',6:'potato',7:'tomato'}

#maxi=max(crop_predict[0])

predicted_crops=[]
temp=list(crop_predict[0])
for i in range(3):
    maxi=max(temp)
    if maxi>0.3:
        predicted_crops.append(maxi)
        temp.remove(maxi)
    else:
        if len(predicted_crops)<2:
            predicted_crops.append(maxi)
            break
        
temp_list=list(crop_predict[0])

predicted_crops_list=[]
for i in predicted_crops:
    predicted_crops_list.append(crops_r[temp_list.index(i)])

'''for i in range(len(crop_predict[0])):
    if crop_predict[0][i] in predicted_crops:
        print(i)
        predicted_crops_list.append(crops_r[i])'''

j=1
output=''    
for i in predicted_crops_list:
    output=output+str(j)+'.'+i+'  '
    j+=1
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))