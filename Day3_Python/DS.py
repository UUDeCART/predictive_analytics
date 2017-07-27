
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV,ElasticNetCV,LassoCV,RidgeCV,LinearRegression
from sklearn import svm, naive_bayes, neighbors
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor, export_graphviz
from sklearn.feature_selection import chi2, f_regression, SelectKBest as skb
from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error as mse
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split as tts, StratifiedShuffleSplit as sss, cross_val_score as cvs, KFold as kf


import warnings
warnings.filterwarnings("ignore")


filesReg=["diabetes","parkinsons"]
filesBinClass=["PimaAmericanIndianDiabetes","Heart","ThoracicSurgery","ICUMIMIC"]
filesMultiClass=["Thyroid"]
files={'filesReg':filesReg,'filesBinClass':filesBinClass,'filesMultiClass':filesMultiClass}
regsNames=['LinearRegression','RidgeCV','LassoCV','ElasticNetCV']

Regs=[LinearRegression(normalize=True), RidgeCV(alphas=[0.1,0.3,0.5,0.7,1.0,3.0,5.0,7.0,10.0],cv=10,normalize=True),LassoCV(alphas=[0.1,0.3,0.5,0.7,1.0,3.0,5.0,7.0,10.0],cv=10,normalize=True), ElasticNetCV(alphas=[0.1,0.3,0.5,0.7,1.0,3.0,5.0,7.0,10.0],cv=10,normalize=True)]
Classifiers=[LogisticRegressionCV(cv=10),DecisionTreeClassifier(max_depth=3),svm.SVC(kernel='rbf',probability=True), svm.SVC(kernel='linear',probability=True), neighbors.KNeighborsClassifier(n_neighbors=7)]
             #naive_bayes.GaussianNB(),neighbors.KNeighborsClassifier(n_neighbors=7)]
ClassifiersNames=['LogisticRegressionCV','DecisionTreeClassifier','svm.SVC_rbf','svm.SVC_linear','neighbors.KNeighborsClassifier']

RegsComp=[LinearRegression(normalize=True), DecisionTreeRegressor(max_depth=3), svm.SVR(kernel='rbf'),neighbors.KNeighborsRegressor(n_neighbors=7)]
RegsCompNames=['LinearRegression', 'DecisionTreeRegressor', 'svm.SVR','KNeighborsRegressor']

tutorialPath='/home/jovyan/DATA/'
clinicalDataPath=tutorialPath+'PA_Data/'


def main ():
     #currentOS=os.getcwd()
     #os.chdir(currentOS+'/data/')
     userInput=input("Select the choice \n (1) Load data regression \n (2) Load data binary classification \n (3) Load data multi-classification \n (4) Describe data \n (5) Mean/SD of numeric data \n (6) Frequencies of categorical data \n (7) Data encoding \n (8) Data scaling \n (9) Find the best Regression on synthesis data  \n (10)Feature selection for regression  \n (11)Run regression on single dataset with Kfold \n (12)Run multiple regression models on multiple datasets with Kfold \n (13)Run linear regression model on multiple datasets using single train/val dataset \n (14)Learning curves of multiple regression models on multiple datasets using single train/val dataset \n (15)Best Classifiers on multiple datasets using Kfold \n (16)Feature selection for classification \n (17)Decision Tree for datasets \n (18)Confusion matrices of classifiers for multiple datasets using single train/val \n (19)Logistic regression for datasets \n (20) Comparison regression models for datasets using Kfold \n") 
     if (userInput=='1'):
         userInput=input("Enter the Index of Dataset 0-Diabetes/1-Parkinsons \n")
         return load_data(int(userInput),'filesReg')
     elif (userInput=='2'):
         userInput=input("Enter the Index of Dataset 0-PimaAmericanIndianDiabetes/1-Heart/2-ThoracicSurgery/3-ICUMIMIC \n")
         return load_data(int(userInput),'filesBinClass')
     elif (userInput=='3'):
         userInput=input("Enter the Index of Dataset 0-Thyroid\n")
         return load_data(int(userInput),'filesMultiClass')     
     elif (userInput=='4'):
         userInput=input("Enter the Index of Dataset 0-Diabetes/1-Parkinsons \n")
         return numDes(int(userInput),'filesReg')
     elif (userInput=='5'):
         userInput=input("Enter the Index of Dataset 0-Diabetes/1-Parkinsons \n")
         meanSDDes(int(userInput),'filesReg')
     elif (userInput=='6'):
         userInput=input("Enter the Index of Dataset 0-PimaAmericanIndianDiabetes/1-Heart/2-ThoracicSurgery/3-ICUMIMIC \n")
         freqNomVars(int(userInput),'filesBinClass')
     elif (userInput=='7'):
         userInput=userInput=input("Enter the Index of Dataset 0-PimaAmericanIndianDiabetes/1-Heart/2-ThoracicSurgery/3-ICUMIMIC \n")
         return dataEncoding(int(userInput),'filesBinClass')
     elif (userInput=='8'):
         userInput=input("Enter the Index of Dataset 0-Diabetes/1-Parkinsons \n")
         return dataScaling(int(userInput),'filesReg')
     elif (userInput=='9'):
        return bestRegressionSynthesis(Regs,[0,1,2])
     elif (userInput=='10'):
         userInput=input("Enter the Index of Dataset 0-Diabetes/1-Parkinsons \n")
         return uniFeatureReg(int(userInput),'filesReg')
     elif (userInput=='11'):
         return runBestRegressionModelKFold([1],[Regs[0]],[0])
     elif (userInput=='12'):
         return runBestRegressionModelKFold([0,1],Regs,list(range(0,len(Regs))))
     elif (userInput=='13'):
         runDifferentRegressionTrTeDataSet([0,1],Regs,list(range(0,len(Regs))))
     elif (userInput=='14'):
         plotLearningCurvesRegression([0,1],[Regs[0]],[0])
     elif (userInput=='15'):
         return runBestClassificationKFold([0,1,2,3],Classifiers,list(range(0,len(Classifiers))))
     elif (userInput=='16'):
         userInput=userInput=input("Enter the Index of Dataset 0-PimaAmericanIndianDiabetes/1-Heart/2-ThoracicSurgery/3-ICUMIMIC \n")
         return uniFeatureChi2(int(userInput),'filesBinClass')
     elif (userInput=='17'):
         return runDTOneTrTe([0,1,2,3])
     elif (userInput=='18'):
         return  runClassifierConfussionMatrix([0,1,2,3],Classifiers,list(range(0,len(Classifiers))))
     elif (userInput=='19'):
         return runLROneTrTe([0,1,2,3])    
     elif (userInput=='20'):
         return runBestRegsCompKFold([0,1],RegsComp)

    
     #os.chdir(currentOS)

def load_data(index=0,taskID='filesBinClass'):
     
        if (taskID=='filesBinClass'):
            csv_path=os.path.join(clinicalDataPath, files['filesBinClass'][index]+".csv")
        elif (taskID=='filesMultiClass'):
            csv_path=os.path.join(clinicalDataPath, files['filesMultiClass'][index]+".csv")
        elif (taskID=='filesReg'):
            csv_path=os.path.join(clinicalDataPath, files['filesReg'][index]+".csv")
        else:
            return "File Reading Error","",""
        myPD=pd.read_csv(csv_path)
        predictors=myPD.columns[:len(myPD.columns)-1]
        target=myPD.columns[len(myPD.columns)-1]
        myTrain=myPD[predictors]
        myVal=myPD[target]
        
        return   myPD,myTrain, myVal


def numDes(index=0,taskID='filesReg'):
    _,myTrain,myVal=load_data(index,taskID)
    myResults={}    
    myTrain.hist(bins=50, figsize=(20,15))  
    myResults['myTrain']= myTrain.describe()
    myResults['myVal']= myVal.describe()
    return myResults

def meanSDDes(index=0,taskID='filesReg'):
    _,myTrain,myVal=load_data(index,taskID)
         
    for name in myTrain.columns:
        if (not(myTrain[name].dtype=='O')):
            print('The mean/SD of ',name,' are ', myTrain[name].mean(),'/',myTrain[name].std())
      
    if (not(myVal.dtype=='O')):
        print('The mean/SD of ',myVal.name,' are ', myVal.mean(),'/',myVal.std())
    
    input('Press Any Key...')
        

def freqNomVars(index=0,taskID='filesReg'):
    _,myTrain,myVal=load_data(index,taskID)
    
    for name in myTrain.columns:
        if (myTrain[name].dtype=='O'):
            print('The frequencies of ',name,' is \n ', myTrain[name].value_counts())
            input('Press Any Key...')
        
    if (myVal.dtype=='O'):
        print('The frequencies of ',myVal.name,' is \n ', myVal.value_counts())
    
def dataEncoding(index=0,taskID='filesReg'):
    le=pre.LabelEncoder()
    data,myTrain,myVal=load_data(index,taskID)
    for name in myTrain.columns:
        if (myTrain[name].dtype=='O'):
            le.fit(myTrain[name])
            myTrain[name]=le.transform(myTrain[name])            
    return data,myTrain,myVal

def dataScaling(index=0,taskID='filesReg'):
    data,myTrain,myVal=load_data(index,taskID)
    for name in myTrain.columns:
        if (not(myTrain[name].dtype=='O')):
            myTrain[name]=pre.minmax_scale(myTrain[name].astype('float'))             
    return data,myTrain,myVal

def uniFeatureReg(index=0,taskID='filesReg'):
    _,myTrain,myVal=dataEncoding(index,taskID)
    
    for name in myTrain.columns:
        if (not(myTrain[name].dtype=='O')):
           myTrain[name]=pre.minmax_scale(myTrain[name].astype('float'))
    return f_regression(myTrain,myVal)[1]

def uniFeatureChi2(index=0,taskID='filesBinClass'):
    _,myTrain,myVal=dataEncoding(index,taskID)
    
    return chi2(myTrain,myVal)[1]
           
def bestRegressionSynthesis(Regs=[],names=[]): 
    
    np.random.seed(42)
    m = 100
    X =  pre.scale(100* np.random.rand(m, 1) - 3)
    y =  0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    
    X_train, X_val, y_train, y_val = tts(X, y, test_size=0.2, random_state=42)
    
    infinity = float("inf")
    index=-1
    count =-1
    
    for reg in Regs:
        count = count +1
        reg.fit(X_train, y_train.ravel())
        pred=reg.predict(X_val)
        meanSquareRootError=np.sqrt(mse(pred,y_val.ravel()))
        print(regsNames[names[count]],meanSquareRootError)
        if (meanSquareRootError < infinity):
            infinity = meanSquareRootError
            index = count
            L1,L2,L3,L4,L5= reg.intercept_,reg.coef_,pred, np.exp(reg.coef_), meanSquareRootError
    return regsNames[names[index]],L1,L2,L3,L4,L5

def runDifferentRegressionTrTeDataSet(dataSets=[],regModels=[],names=[]): 
    
    myResults={}
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        for name in myTrain.columns:
          if (not(myTrain[name].dtype=='O')):
            myTrain[name]=pre.minmax_scale(myTrain[name].astype('float')) 
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.20, random_state=42)
        infinity = float("inf")
        index=-1 
        count =-1
        for reg in regModels:
            count = count +1
            reg.fit(X_train, y_train.ravel())
            pred=reg.predict(X_val)
            meanSquareRootError=np.sqrt(mse(pred,y_val.ravel()))
            print(regsNames[names[count]],meanSquareRootError)
            if (meanSquareRootError < infinity):
                infinity = meanSquareRootError
                index = count
                L1,L2,L3,L4,L5,L6= regsNames[names[index]],reg.intercept_,reg.coef_, np.exp(reg.coef_), pred,infinity

            plt.plot(pred, y_val,'ro')
            plt.plot([0,300],[0,300], 'g-')
            plt.xlabel('predicted')
            plt.ylabel('real')
            plt.show()
            input('Press any key')
        print(filesReg[ds],regsNames[names[index]],infinity)
        myResults[filesReg[ds]]={1:L1,2:L2,3:L3,4:L4,5:L5,6:L6}
        print('\n')     
    return myResults    
    
def runBestRegressionModelKFold(dataSets=[],regModels=[],names=[]):

    myResults={}
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        for name in myTrain.columns:
          if (not(myTrain[name].dtype=='O')):
            myTrain[name]=pre.minmax_scale(myTrain[name].astype('float')) 
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        splits =kf(n_splits=10, shuffle=True, random_state=42)
        infinity = float("inf")
        index=-1 
        count =-1
        for reg in regModels:
            count = count +1
            reg.fit(myTrain, myVal)
            cvsScores=cvs(reg, myTrain, myVal,cv=splits,scoring='neg_mean_squared_error')
            meanSquareRootError=np.sqrt(-1*cvsScores.mean())
            print(regsNames[names[count]],meanSquareRootError)
            if (meanSquareRootError < infinity):
                infinity = meanSquareRootError
                index = count
                L1,L2,L3,L4,L5,L6= regsNames[names[index]],reg.intercept_,reg.coef_, np.exp(reg.coef_), cvsScores, infinity
        print(filesReg[ds],regsNames[names[index]],infinity)
        myResults[filesReg[ds]]={1:L1,2:L2,3:L3,4:L4,5:L5,6:L6}
        print('\n')     
    return myResults    

def plotLearningCurvesRegression(dataSets=[],regModels=[],names=[]):
 
    
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        for name in myTrain.columns:
          if (not(myTrain[name].dtype=='O')):
            myTrain[name]=pre.minmax_scale(myTrain[name].astype('float')) 
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.2, random_state=42)
        count =-1
        for reg in regModels:
            count = count+1
            train_errors,val_errors=[],[]
            for m in range(1, len(X_train)):
                reg.fit(X_train[:m], y_train[:m])
                y_train_predict = reg.predict(X_train[:m])
                y_val_predict = reg.predict(X_val)
                train_errors.append(mse(y_train_predict, y_train[:m]))
                val_errors.append(mse(y_val_predict, y_val))
            plt.plot(np.sqrt(train_errors), "r-+", linewidth=4, label="train")
            plt.plot(np.sqrt(val_errors), "b-", linewidth=4, label="val")
            plt.legend(loc="upper right", fontsize=14)    
            plt.xlabel("Training set size", fontsize=14)  
            plt.ylabel("MSE", fontsize=14)
            plt.show()
            print(filesReg[ds],regsNames[names[count]])
            input("Press Any Key")


def runBestClassificationKFold(dataSets=[],Classifiers=[],names=[]):

    myResults={}
    le=pre.LabelEncoder()
    
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesBinClass')    
        le.fit(myVal)
        myVal=le.transform(myVal)
        #myTrain = skb(f_regression, k=6).fit_transform(myTrain,myVal)
        #myTrain = skb(chi2, k=5).fit_transform(myTrain,myVal)
        splits = sss(n_splits=10, test_size=((len(myData)*.20)/len(myData)), random_state=42)
        #splits =kf(n_splits=10, shuffle=True, random_state=42)
        infinity = -1.0 * float("inf")
        index=-1 
        count =-1
        for clf in Classifiers:
            count = count +1
            clf.fit(myTrain, myVal)
            cvsScores=cvs(clf, myTrain, myVal,cv=splits,scoring='roc_auc')
            meanAUC=cvsScores.mean()
            print(ClassifiersNames[names[count]],meanAUC)
            if (meanAUC > infinity):
                infinity = meanAUC
                index = count
                L1,L2,L3 = ClassifiersNames[names[index]],cvsScores, infinity
        print(filesBinClass[ds],ClassifiersNames[names[index]],infinity)
        myResults[filesBinClass[ds]]={1:L1,2:L2,3:L3}
        print('\n')     
    return myResults  

def runDTOneTrTe(dataSets=[]):
    
    Results={}
    for ds in dataSets:
        _,myTrain,myVal=dataEncoding(ds,taskID='filesBinClass')
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.2, random_state=42)
        
        Classifiers[1].fit(X_train,y_train)
        with open('DTclassifier_'+filesBinClass[ds]+'.txt', "w") as f:
            f = export_graphviz(Classifiers[1], 
                                class_names=y_val.name,
                                feature_names= myTrain.columns,
                                rounded=True,
                                filled=True,
                                out_file=f)
        
        Results[ds]= Classifiers[1].score(X_val,y_val)
   
    return Results


def runClassifierConfussionMatrix(dataSets=[],Classifiers=[],names=[]):
    
    Results={}
    for ds in dataSets:
        _,myTrain,myVal=dataEncoding(ds,taskID='filesBinClass')
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.20, random_state=42)
        count =-1
        cls={}
        for clf in Classifiers:
            count = count +1
            Classifiers[count].fit(X_train,y_train)
            pred =Classifiers[count].predict(X_val)
            tn, fp, fn, tp=confusion_matrix(y_val,pred).ravel()
            cls[ClassifiersNames[names[count]]]= [tn, fp, fn, tp]
             
        Results[filesBinClass[ds]]= cls
   
    return Results

def runLROneTrTe(dataSets=[]):
    
    Results={}
    for ds in dataSets:
        _,myTrain,myVal=dataEncoding(ds,taskID='filesBinClass')
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.2, random_state=42)
        
        Classifiers[0].fit(X_train,y_train)
        pred =Classifiers[0].predict(X_val)
        Results[filesBinClass[ds]]=[Classifiers[0].intercept_,Classifiers[0].coef_,pred, np.exp(Classifiers[0].coef_),accuracy_score(y_val, pred)]
        
    return Results

def runBestRegsCompKFold(dataSets=[],regModels=[],names=[]):
    
    myResults={}
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        for name in myTrain.columns:
          if (not(myTrain[name].dtype=='O')):
            myTrain[name]=pre.minmax_scale(myTrain[name].astype('float')) 
        splits =kf(n_splits=10, shuffle=True, random_state=42)
        infinity = float("inf")
        index=-1 
        count =-1
        for reg in regModels:
            count = count +1
            reg.fit(myTrain, myVal)
            cvsScores=cvs(reg, myTrain, myVal,cv=splits,scoring='neg_mean_squared_error')
            meanSquareRootError=np.sqrt(-1*cvsScores.mean())
            print(RegsCompNames[names[count]],meanSquareRootError)
            if (meanSquareRootError < infinity):
                infinity = meanSquareRootError
                index = count
                L1,L2,L3= RegsCompNames[names[index]],cvsScores, infinity
        print(filesReg[ds],RegsCompNames[names[index]],infinity)
        myResults[filesReg[ds]]={1:L1,2:L2,3:L3}
        print('\n')     
    return myResults 

def indexToyExample():
     
     dataDict={'Age':list(np.random.uniform(low=30.0,high=79.0,size=1000)),'Sex':list(np.random.randint(2,size=1000)), 'SBP':list(np.random.uniform(low=90.0,high=180.0,size=1000)),'Smoker':list(np.random.randint(2,size=1000)),'CHF':list(np.random.randint(2,size=1000))}
     myPD=pd.DataFrame(dataDict)
     
     predictors=['Age','Sex','SBP']
     target=['CHF']
     

     myTrain=myPD[predictors]
     newMyTrain= myPD[predictors]
     myVal=myPD[target]
     
     splits =kf(n_splits=10, shuffle=True, random_state=42)
     LR=Classifiers[0].fit(myTrain,myVal) 
     cvsScores=cvs(LR, myTrain, myVal,cv=splits,scoring='neg_mean_squared_error')
     LR.predict(myTrain)
     meanSquareRootError=np.sqrt(-1*cvsScores.mean())
     L1= {1:LR.intercept_,2:LR.coef_, 3:np.exp(LR.coef_), 4:cvsScores, 5:meanSquareRootError}
     
    
     

     LS=[]
     LY=[]
     for index, row in myTrain.iterrows():
         if (row['Age'] < 39):
             LS.append(1) 
         elif (row['Age'] <= 49):
             LS.append(2) 
         elif (row['Age'] <= 59):
             LS.append(3)
         elif (row['Age'] <= 69):
             LS.append(4)
         elif (row['Age'] <= 80):
             LS.append(5)
    
         if (row['SBP'] < 120):
             LY.append(1)
         elif (row['SBP'] < 129):
             LY.append(2)
         elif (row['SBP'] < 139):
             LY.append(3)
         elif (row['SBP'] < 159):
             LY.append(4)
         else:
             LY.append(5) 
    
    
     newMyTrain['Age'] = LS
     newMyTrain['SBP']=LY
     
     newMyPD=pd.DataFrame()
     

     newMyPD['Age']= newMyTrain['Age']
     newMyPD['Sex']= newMyTrain['Sex']
     newMyPD['SBP']= newMyTrain['SBP']
     newMyPD['CHF']=myVal

     Freq={'Age':newMyPD.groupby( [ 'Age','CHF'] ).size(), 'Sex':newMyPD.groupby( [ 'Sex','CHF'] ).size(), 'SBP':newMyPD.groupby( [ 'SBP','CHF'] ).size()}
         
     return   myPD,myTrain, myVal,newMyPD,L1,Freq
     
def votingClassifiers():
    # This example is taken from https://github.com/ageron/handson-ml/blob/master/07_ensemble_learning_and_random_forests.ipynb 


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    

    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42)

    _,myTrain,myVal=dataEncoding(0,'filesBinClass')
    X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.2, random_state=42)

    voting_clf = VotingClassifier(
            estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
            voting='hard')
     
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(clf.__class__.__name__, accuracy_score(y_val, y_pred))

def baggingClassifiers():
    # This example is taken from https://github.com/ageron/handson-ml/blob/master/07_ensemble_learning_and_random_forests.ipynb 


   from sklearn.ensemble import BaggingClassifier
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score
   
   _,myTrain,myVal=dataEncoding(2,'filesBinClass')
   X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.2, random_state=42)

   bag_clf = BaggingClassifier(
            DecisionTreeClassifier(random_state=42), n_estimators=500,
            max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
   bag_clf.fit(X_train, y_train)
   y_pred = bag_clf.predict(X_val)
   print(accuracy_score(y_val, y_pred))
     

   tree_clf = DecisionTreeClassifier(random_state=42)
   tree_clf.fit(X_train, y_train)
   y_pred_tree = tree_clf.predict(X_val)
   print(accuracy_score(y_val, y_pred_tree))

def BoostingRegressors():
    # This example is taken from https://github.com/ageron/handson-ml/blob/master/07_ensemble_learning_and_random_forests.ipynb 


   from sklearn.tree import DecisionTreeRegressor 
   from sklearn.metrics import mean_squared_error
   
   _,myTrain,myVal=dataEncoding(0,'filesReg')
   X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.2, random_state=42)

   tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
   tree_reg1.fit(X_train, y_train)
   y_pred = tree_reg1.predict(X_val)
   print(np.sqrt(mean_squared_error(y_val, y_pred)))
   
   y2 = y_train - tree_reg1.predict(X_train)
   tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
   tree_reg2.fit(X_train, y2)
   y_pred = sum(tree.predict(X_val) for tree in (tree_reg1, tree_reg2))
   print(np.sqrt(mean_squared_error(y_val, y_pred)))

 
   y3 = y2 - tree_reg2.predict(X_train)
   tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
   tree_reg3.fit(X_train, y3)
   y_pred = sum(tree.predict(X_val) for tree in (tree_reg1, tree_reg2, tree_reg3))
   print(np.sqrt(mean_squared_error(y_val, y_pred)))   
   

def runBestRegressionModelKFoldwFS(dataSets=[],regModels=[],names=[]):

    myResults={}
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        myTrain = skb(f_regression, k=5).fit_transform(myTrain,myVal)
        splits =kf(n_splits=10, shuffle=True, random_state=42)
        infinity = float("inf")
        index=-1 
        count =-1
        for reg in regModels:
            count = count +1
            reg.fit(myTrain, myVal)
            cvsScores=cvs(reg, myTrain, myVal,cv=splits,scoring='neg_mean_squared_error')
            meanSquareRootError=np.sqrt(-1*cvsScores.mean())
            print(regsNames[names[count]],meanSquareRootError)
            if (meanSquareRootError < infinity):
                infinity = meanSquareRootError
                index = count
                L1,L2,L3,L4,L5,L6= regsNames[names[index]],reg.intercept_,reg.coef_, np.exp(reg.coef_), cvsScores, infinity
        print(filesReg[ds],regsNames[names[index]],infinity)
        myResults[filesReg[ds]]={1:L1,2:L2,3:L3,4:L4,5:L5,6:L6}
        print('\n')     
    return myResults


def runBestRegressionModelKFoldPrintFolderErrors(dataSets=[],regModels=[],names=[]):

    myResults={}
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        splits =kf(n_splits=2, shuffle=True, random_state=42)
        infinity = float("inf")
        index=-1 
        count =-1
        for reg in regModels:
            count = count+1
            xval_err = 0
            for train,test in splits.split(myTrain):
                reg.fit(myTrain.ix[train],myVal.ix[train])
                p = reg.predict(myTrain.ix[test])
                e = p-myVal.ix[test]
                print (e)
                xval_err += np.dot(e,e)
            rmse_10cv = np.sqrt(xval_err/len(myTrain))
            print(rmse_10cv)
            input("Press any key")
            if (rmse_10cv < infinity):
                infinity = rmse_10cv
                index = count
                L1,L2,L3,L4,L5= regsNames[names[index]],reg.intercept_,reg.coef_, np.exp(reg.coef_), infinity
        print(filesReg[ds],regsNames[names[index]],infinity)
        myResults[filesReg[ds]]={1:L1,2:L2,3:L3,4:L4,5:L5}
        print('\n')     
    return myResults

def runRidgeRegressiontoEstAlpha(dataSets=[]):

    from sklearn.linear_model import Ridge
    
    
    print('Ridge Regression')
    print('alpha\t RMSE_train\t RMSE_10cv\n')
    alpha = np.linspace(.01,20,50)

    
    for ds in dataSets:
        myData,myTrain,myVal=dataEncoding(ds,taskID='filesReg')
        #for name in myTrain.columns:
          #if (not(myTrain[name].dtype=='O')):
            #myTrain[name]=pre.minmax_scale(myTrain[name].astype('float')) 
        t_rmse = np.array([])
        cv_rmse = np.array([])
        
        for a in alpha:
            ridge = Ridge(fit_intercept=True, alpha=a)
            ridge.fit(myTrain,myVal)
            p = ridge.predict(myTrain)
            err = p-myVal
            total_error = np.dot(err,err)
            rmse_train = np.sqrt(total_error/len(p))
    
            splits =kf(n_splits=10, shuffle=True, random_state=42)
            xval_err = 0
            for train,test in splits.split(myTrain):
                ridge.fit(myTrain.ix[train],myVal.ix[train])
                p = ridge.predict(myTrain.ix[test])
                e = p-myVal.ix[test]
                xval_err += np.dot(e,e)
            rmse_10cv = np.sqrt(xval_err/len(myTrain))     
    
            t_rmse = np.append(t_rmse, [rmse_train])
            cv_rmse = np.append(cv_rmse, [rmse_10cv])
            print('{:.3f}\t {:.4f}\t\t {:.4f}'.format(a,rmse_train,rmse_10cv))
        input("Press Any Key")
        
    plt.plot(alpha, t_rmse, label='RMSE-Train')
    plt.plot(alpha, cv_rmse, label='RMSE_CV')
    plt.legend( ('RMSE-Train', 'RMSE_XCV') )
    plt.ylabel('RMSE')
    plt.xlabel('Alpha')
    plt.show()

def compareAUCwithROC(dataSets=[],Classifiers=[],names=[]):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    
    clf_labels=[]
    le=pre.LabelEncoder()
    
    for n in names:
        clf_labels.append(ClassifiersNames[n])
    
    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']    
    
    

    for ds in dataSets:
        _,myTrain,myVal=dataEncoding(ds,taskID='filesBinClass')
        le.fit(myVal)
        myVal=le.transform(myVal)
        
        #myTrain = skb(f_regression, k=3).fit_transform(myTrain,myVal)
        X_train, X_val, y_train, y_val = tts(myTrain,myVal, test_size=0.20, random_state=42)

        
        for clf, label, clr, ls  in zip(Classifiers, clf_labels, colors, linestyles):
            y_pred = clf.fit(X_train,y_train).predict_proba(X_val)[:, 1] # Assume 1 is the positive class
            fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_pred)
            roc_auc = auc(x=fpr, y=tpr)
            plt.plot(fpr, tpr,color=clr,linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()    

