import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
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


    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X, y)
        

    int_features = [x for x in request.form.values()]
    
    
    crops={'carrot':0,'potato':1,'beans':2,'cauliflower':3,'bottleguard':4,'tomato':5,'cowpeas':6,'chillies':7}
    
    crops_list=int_features

    arr=np.array([0,0,0,0,0,0,0,0])
    for i in crops_list:
        arr[crops[i]]+=1
    
    '''output=''
    for i in arr:
        output+=str(i)'''
    '''arr=np.array([0,0,0,0])
    for i in crops_list:
        arr[crops[i]]<1:
            arr[crops[i]]+=1'''
    
    
    crop_predict=regressor.predict(np.array([arr]))
    crops_r={0:'beans',1:'bottleguard',2:'carrot',3:'cauliflower',4:'chillies',5:'cowpeas',6:'potato',7:'tomato'}

    #crops_r={3:'wheat',0:'garlic',1:'rice',2:'tomato'}
    
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
    
    '''maxi=max(crop_predict[0])
    predicted_crop=''
    
    if maxi>0.7:
        for i in range(len(crop_predict[0])):
            if crop_predict[0][i]==maxi:
                predicted_crop=crops_r[i]'''
                
    temp_list=list(crop_predict[0])
    predicted_crops_list=[]
    for i in predicted_crops:
        predicted_crops_list.append(crops_r[temp_list.index(i)])
    
    '''predicted_crops_list=[]
    for i in range(len(crop_predict[0])):
        if crop_predict[0][i] in predicted_crops:
            predicted_crops_list.append(crops_r[i])'''
    

    j=1
    output=''    
    for i in predicted_crops_list:
        output=output+str(j)+'.'+i+'  '
        j+=1
    
    #output = predicted_crop

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted crop are: '+output)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)