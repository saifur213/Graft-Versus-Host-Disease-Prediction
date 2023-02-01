# Graft versus Host Disease Prediction

Allogenic stem cell transplantation (ASCT) is 
an effective surgical treatment that is widely used
for hematologic malignancies and other blood disorders.
Unfortunately, Graft versus Host Disease (GvHD) is a 
typical adverse effect of ASCT. In this project, we 
created an ML model that can predict a patient wheater he
or she will get affected by GvHD or not after an ASCT.

## Authors

- [@Md_Asif_Binkhaled](https://gitlab.com/mdasifbinkhaled)
- [@Md_Junayed_Hossain](https://gitlab.com/Junayed7166)
- [@Md_Saifur_Rahman](https://gitlab.com/saifur15)
- [@Jannatul_Ferdaus](https://gitlab.com/Jannatul.04)
 


## Installation

Installation of required library.

!pip install chord  
!pip install missingno  
!pip install missingpy  
!pip install soccerplots  
!pip install mplsoccer

```bash
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.style as style
from sklearn.metrics import confusion_matrix
from sklearn import tree , metrics, preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn import preprocessing

**Machine Learning Models**

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

```
## Road Map of this Project
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/images.png)

##  Dataset source 
[Dataset Source](https://archive.ics.uci.edu/ml/datasets/Bone+marrow+transplant%3A+children)

## Numerical Features
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/numerical.png)

## Categorical Features
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/categorical.png)

## Missing Values
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/missing_values.png)

## Imputing Missing Values
Missing values were imputed in two different ways. They are:<br />
    1. **Miss Forest algorithm :** To impute numerical missing values.<br />
    2. **Random Forest Classifier :** To predict the categorical missing values.<br />

## Balancing the Dataset
**The was balanced using SMOTE (Synthetic Minority Over-sampling Technique)**<br />
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/Balance.png)

## Feature Engineering
**correlation matrix** <br /> 
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/Corellation.png)

**Selected Features for Performing Machine Learning Algorithms**<br />
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/Distplot.png)

## Result 
![alt text](https://gitlab.com/mdasifbinkhaled/graft_versus_host_disease_prediction/-/raw/main/Images/Result_bg_white.png)
