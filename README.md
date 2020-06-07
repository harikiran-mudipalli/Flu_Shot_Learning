# Flu_Shot_Learning
 Competiton hosted by : DRIVEDATA at https://www.drivendata.org/competitions/66/flu-shot-learning/ <br>
 **Goal** : Can you predict whether people got H1N1 and seasonal flu vaccines using information they shared about their     backgrounds, opinions, and health behaviors? Your goal is to predict how likely individuals are to receive their H1N1 and seasonal flu vaccines. Specifically, you'll be predicting two probabilities: one for ```h1n1_vaccine``` and one for ```seasonal_vaccine```.<br>
 **Metric** : AUC_ROC
 <br>
 ### model.py<br>
 ```
 import model
 model.fit_and_estimate(features, target, estimator, label=None, multilabel=False, scale=True, scaler=MinMax)
 ```
 parameters:<br>
 features - Independent features<br>
 target - Dependent features<br>
 estimator - classification model. If the ```multilabel=True``` use ```MultiOutputClassifier()``` along with classifier<br>
 multilabel - default: False. If you want to train on more than one target variable, make ```multilabel=True```.<br>
 label - default: False. If you want to train on each target variable individually, the specify the label to be trained on. Example: ```label='h1n1_vaccine'```.<br>
 scale - default: True. Set ```scale=False``` if you wish not to scale the inputs to the classifier.<br>
 scaler - default: ```StandardScaler()```. Other scaling option available is ```MinMaxScaler()```. Example ```scaler="MinMax"```.<br>
 ### tune.py<br>
 ```
 from tune import ModelTune
 estimator = ModelTune(X, y).logreg()
 ```
 parameters:<br>
 X - Independent features.<br>
 y - single target variable.<br>
 <br>
 Available classifiers:
 <br>
 Logistic Regression - ```logreg()```<br>
 Gradient Boosting Classifier - ```GBC()```<br>
 K neighbors Classifier - ```knn()```<br>
 Decision Tree Classifer - ```dtc()```<br>
 Random Forest Classifer - ```rfc()```<br>
 Ada Boost Classifer - ```ada()```<br>
