# Create your views here.
import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse,HttpResponseRedirect
'''import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import numpy as np
#from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from mlxtend.plotting import category_scatter
from sklearn.metrics import confusion_matrix 
from mlxtend.plotting import plot_confusion_matrix
from pylab import *
import base64,urllib,io
from io import StringIO, BytesIO
import importlib


#matplotlib = importlib.reload(matplotlib)
#matplotlib.use('cairo')
import matplotlib.pyplot as plt,mpld3
'''





import xlrd
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder








# Create your views here.
crop2='jj'
from .forms import InputForm
#dataframe = pandas.read_csv('static/pima_indians_diabetes.csv')

def get_text(request):
#    dataframe = pandas.read_csv('static/pima_indians_diabetes.csv')
    #print(dataframe.head())
    if request.method == 'POST':
        form = InputForm(request.POST)
        request.session['arr']=[]
        
        if form.is_valid():
           # a1=request.session['Text'] = form.cleaned_data['Text']
            #arr.append(int(a1))
            request.session['temp'] = int(form.cleaned_data['temp'])
        #    arr.append(int(a1))
            request.session['rainfall'] = int(form.cleaned_data['rainfall'])
         #   arr.append(int(a2))
            request.session['soil'] = int(form.cleaned_data['soil'])
          #  arr.append(int(a3))
            request.session['crop'] = int(form.cleaned_data['crop'])
            global crop2
            crop2=int(form.cleaned_data['crop'])
           # arr.append(int(a4))
            request.session['land'] = int(form.cleaned_data['land'])
            #print(arr)            
            request.session['arr'].append([request.session['temp'],request.session['rainfall'],
                request.session['soil'],request.session['crop'],request.session['land']])
            print(request.session['arr'])

            return HttpResponseRedirect('/result/')
    else:
        form = InputForm()

    return render(request,'input.html', {'form': form})
'''

======
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)
=====
'''


#===================================================
def result(request):
    
    
    
    data_train = pd.read_csv('regressiondb.csv')
    data_test = pd.read_excel('finalr.xlsx')

 
    print("entered in to request")
    label_encoder = LabelEncoder()
    temp = data_train['Crop']
    temp = array(temp)
    encoded = label_encoder.fit_transform(temp)
    temp1 = data_test['Crop']
    temp1 = array(temp1)
    encoded1 = label_encoder.transform(temp1)
    data_test['Crop'] = encoded1
    data_train['Crop'] = encoded
    x_train = data_train.iloc[:,0:4]
    x_train = array(x_train)
    y_train = data_train.iloc[:,4]
    y_train = array(y_train)
    x_test = data_test.iloc[:,0:4]
    y_test = data_test.iloc[:,4]
    x_test = array(x_test)
    y_test = array(y_test)
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)

    k = regressor.score(x_test,y_test)
    r=round(abs(k)*100,2)
    print("the accuracy is :",round(abs(k)*100,2 ),'%')

#reading the values from website stored in arr

    data1 = (request.session['arr'])
    print(data1[0][4])
    land1=data1[0][4]
    rainfalll=data1[0].pop(1)
    data= np.array(data1,ndmin=2)
    print("data",data)
    test_result=0
    df = pd.DataFrame(data,columns = ['Rainfall','Temperature','Ph','Crop'])
    x = df.iloc[:,:].values
    if(x[0][0]==0 and x[0][1]==0 and x[0][2]==0 and x[0][3]==0):
            
        print("invalid inputs")
        test_result='Invalid Details'

    elif(x[0][0]==0 or x[0][1]==0 or x[0][2] ==0 or x[0][3]==0):

        test_result='Invalid Details'
    else:
        acres=(land1/2.5)
        test_result= round(abs(regressor.predict(x)[0]))
        test_result=round(test_result*acres,2)
        print("acres",acres)
        print("predict value",regressor.predict(x))
        print("final",test_result)

    if(x[0][0]>=12 and x[0][0]<=50):
        pass
    else:
        print("invalid Temperature")
        test_result='Invalid Temperature'

    if(rainfalll>=400 and rainfalll<=1100):
        pass
    else:
        print(x[0][2])
        print("invalid Rain")
        test_result='Not suitable Rainfall condition'

    if(x[0][1]>=3 and x[0][1]<=10):
        pass
    else:
        print(x[0][1])
        print("invalid Rain")
        test_result='Invalid PH Value'

    #dictionary

    dic={0:'Bajra',1: 'Banana',2: 'Barley', 3:'Bean', 4:'Black pepper', 5:'Blackgram',
       6:'Bottle Gourd', 7:'Brinjal', 8:'Cabbage', 9:'Cardamom', 10:'Carrot',
       11:'Castor seed', 12:'Cauliflower', 13:'Chillies', 14:'Colocosia', 15:'Coriander',
       16:'Cotton', 17:'Cowpea', 18:'Drum Stick', 19:'Garlic', 20:'Ginger', 21:'Gram',
       22:'Grapes', 23:'Groundnut', 24:'Guar seed', 25:'Horse-gram', 26:'Jowar', 27:'Jute',
       28:'Khesari', 29:'Lady Finger', 30:'Lentil', 31:'Linseed', 32:'Maize', 33:'Mesta',
       34:'Moong(Green Gram)', 35:'Moth', 36:'Onion', 37:'Orange', 38:'Papaya',
       39:'Peas & beans (Pulses)', 40:'Pineapple', 41:'Potato', 42:'Raddish', 43:'Ragi',
       44:'Rice', 45:'Safflower', 46:'Sannhamp', 47:'Sesamum', 48:'Soyabean',
       49:'Sugarcane', 50:'Sunflower', 51:'Sweet potato', 52:'Tapioca', 53:'Tomato',
       54:'Turmeric', 55:'Urad', 56:'Varagu', 57:'Wheat'
    }
    
    print(request,r)
    return render(request,'result.html', {'res':r,'test_result':test_result,'p_crop':dic[int(crop2)]})
       # print(":::The crop yielding based on the consider factors",regressor.predict(x),'in tonnes')


        
'''
    print(value)
    print(y_pred1)
    res=accuracy_score(y_test,y_pred)*100
    print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
    cm = confusion_matrix(y_test, y_pred)

     
    fig, ax = plot_confusion_matrix(conf_mat=cm,figsize=(3,3.5))

    #plt.set_size_inches(8.5, 4.5)
  #  grid(True)
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    img2 = urllib.parse.quote(string)


    X = np.array(dataframe[['plas','age']])
    Y1=Y.astype(int)
    y = np.array(Y1)
    neigh.fit(X, y)


# Plotting decision regions
    plt.figure(figsize=(13,7))
    
    plot_decision_regions(X, y, clf=neigh, legend=2)

# Adding axes annotations
    plt.xlabel('Plasma glucose')
    plt.ylabel('Age')
    plt.title('knn classification')
    #grid(True)
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    img3 = urllib.parse.quote(string)
'''



#    return HttpResponse("Thanks")    
     

def abc(request):
    if request.method=='POST':
        s1=request.POST['crop1']
        s2=request.POST['crop2']
        s3=request.POST['crop3']
        
        dataset = pd.read_csv(r"D:\SREENIDHI\crop_yield\crops2.csv")
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 7:9].values
        
        
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder = LabelEncoder()
        y[:,1] = labelencoder.fit_transform(y[:,1])
        onehotencoder = OneHotEncoder( categories = [-1])
        y = onehotencoder.fit_transform(y).toarray()
        y=y[:,:8]
    
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X, y)
        
        int_features = [s1,s2,s3]
        

        crops={'carrot':0,'potato':1,'beans':2,'cauliflower':3,'bottleguard':4,'tomato':5,'cowpeas':6,'chillies':7}
        
        crops_list=int_features
    
        arr=np.array([0,0,0,0,0,0,0,0])
        for i in crops_list:
            arr[crops[i]]+=1
            
        crop_predict=regressor.predict(np.array([arr]))
        crops_r={0:'beans',1:'bottleguard',2:'carrot',3:'cauliflower',4:'chillies',5:'cowpeas',6:'potato',7:'tomato'}

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
        
            print(len(predicted_crops),predicted_crops)
            
        temp_list=list(crop_predict[0])
        predicted_crops_list=[]
        for i in predicted_crops:
            predicted_crops_list.append(crops_r[temp_list.index(i)])
            
        j=1
        output=''    
        for i in predicted_crops_list:
            output=output+str(j)+'.'+i+'  '
            j+=1
        
        print(output)
        return render(request,'index.html',{'prediction_test':'Predicted crops are: '+output})
        
    return render(request,'index.html',{})

def dataset(request):
    return render(request,'dataset.html',{})

def algorithm(request):
    return render(request,'algo.html',{}) 
 




def getimage(request):
    # Construct the graph
    x = arange(0, 2*pi, 0.01)
    s = cos(x)**2
    plot(x, s)

    xlabel('xlabel(X)')
    ylabel('ylabel(Y)')
    title('Simple Graph!')
    grid(True)

    fig = plt.figure()
    plt.plot([3,1,4,1,5])
    figHtml = mpld3.fig_to_html(fig)
    result = {'fileData': myData, 'figure': figHtml}
    #return render_to_response("index.html",result)
    
    # Store image in a string buffer  
    out = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(out, "PNG")
    pylab.close()
    '''


    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = figfile.read()  # extract string
    import base64,urllib,io
    figdata_png = base64.b64encode(figdata_png)
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
#    print(uri)
    
    #uri = 'data:image/png;base64,' + urllib.parse.quote(figdata_png)
    #html = '<img src = "%s"/>' % uri

    return render(request,'img.html',{'uri':uri,'figdata_png':figdata_png},result)
    # Send buffer in a http response the the browser with the mime type image/png set
    #return HttpResponse(out.getvalue(), content_type="image/png")
'''

