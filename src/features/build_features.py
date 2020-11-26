import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_features(data):  # titanic train_data

    data['Alone'] = (data['SibSp'] + data['Parch'] + 1).apply(lambda x: 1 if x ==1 else 0)

    # sex featrue
    Isfemale = pd.get_dummies(data['Sex'])['female']

    # extra features from Age

    # SibSp features
    Isalone = data['Alone'].copy()
    SibSp = data['SibSp'].copy()
    Parch = data['Parch'].copy()

    # class features
    C1 = pd.get_dummies(data['Pclass'])[1]
    C2 = pd.get_dummies(data['Pclass'])[2]

    # numeric features
    scaler = StandardScaler()

    Age = data['Age'].copy()
    Age = pd.Series(scaler.fit_transform(np.array(Age).reshape(-1, 1))[:, 0])

    Fare = data['Fare'].copy()
    Fare = pd.Series(scaler.fit_transform(np.array(Fare).reshape(-1, 1))[:, 0])

    # embarked features
    Isq = pd.get_dummies(data['Embarked'])['Q']
    Iss = pd.get_dummies(data['Embarked'])['S']

    # title features
    IsMr = data['Name'].apply(lambda x: 1 if 'Mr' in x else 0)
    IsMrs = data['Name'].apply(lambda x: 1 if 'Mrs' in x else 0)
    Isminor = pd.get_dummies(data['Title'])[3]
    Ismiss = pd.get_dummies(data['Title'])[2]
    Israre = pd.get_dummies(data['Title'])[4]

    # target feature
    if('Survived' in data.columns): # if data is test, survived is not in df
        y = data['Survived']
    else:
        y = None


    X = pd.concat([Isfemale, Isalone, C1,
                   C2, Age, Fare, Isq, Iss, IsMr, IsMrs,
                   Isminor, Ismiss, Israre, SibSp, Parch], axis=1, ignore_index=True)

    feature_names = ['Isfemale', 'Isalone', 'C1',
                     'C2', 'Age', 'Fare', 'Isq', 'Iss', 'Ismr', 'Ismrs',
                     'Isminor', 'Ismiss', 'Israre', 'SibSp', 'Parch']

    X.columns = feature_names

    return X, y