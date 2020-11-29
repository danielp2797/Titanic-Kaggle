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
    Pclass = data['Pclass']

    # numeric features
    scaler = StandardScaler()

    Age = data['Age'].copy()
    Age = pd.Series(scaler.fit_transform(np.array(Age).reshape(-1, 1))[:, 0])
    # Age in bins
    Age_bins = pd.cut(data['Age'], bins=10, labels=range(10))

    Fare = data['Fare'].copy()
    Fare = pd.Series(scaler.fit_transform(np.array(Fare).reshape(-1, 1))[:, 0])

    Fare_bins = pd.cut(data['Fare'], bins=10, labels=range(10))

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

    Cabin = pd.get_dummies(data['Cabin'].fillna('Missing').apply(lambda x: x[:1]))

    try: # except for test dataframe, T is not in test
        Cabin['M'] = Cabin['M'] + Cabin['T']  # add T Cabin to M
        Cabin.drop('T', axis=1, inplace=True)  # remove Cabin T
    except:
        pass

    X = pd.concat([Isfemale, Isalone, C1,
                   C2, Age, Fare, Isq, Iss, IsMr, IsMrs,
                   Isminor, Ismiss, Israre, SibSp, Parch,
                   Pclass, Age_bins, Fare_bins, Cabin], axis=1, ignore_index=True)

    feature_names = ['Isfemale', 'Isalone', 'C1',
                     'C2', 'Age', 'Fare', 'Isq', 'Iss', 'Ismr', 'Ismrs',
                     'Isminor', 'Ismiss', 'Israre', 'SibSp', 'Parch', 'Pclass',
                     'Age_bins', 'Fare_bins', 'CabinA', 'CabinB',
                     'CabinC', 'CabinD', 'CabinE', 'CabinF', 'CabinG', 'CabinM']

    X.columns = feature_names

    return X, y