import pandas as pd
import numpy as np

def build_features(data):  # titanic train_data

    # sex featrue
    Isfemale = pd.get_dummies(train_data['Sex'])['female']

    # extra features from Age
    Isold = train_data['Old'].copy()

    # SibSp features
    Isalone = train_data['Alone'].copy()
    SibSp = train_data['SibSp'].copy()
    Parch = train_data['Parch'].copy()

    # class features
    C1 = pd.get_dummies(train_data['Pclass'])[1]
    C2 = pd.get_dummies(train_data['Pclass'])[2]

    # numeric features
    scaler = StandardScaler()

    Age = train_data['Age'].copy()
    Age = pd.Series(scaler.fit_transform(np.array(Age).reshape(-1, 1))[:, 0])

    Fare = train_data['Fare'].copy()
    Fare = pd.Series(scaler.fit_transform(np.array(Fare).reshape(-1, 1))[:, 0])

    # embarked features
    Isq = pd.get_dummies(train_data['Embarked'])['Q']
    Iss = pd.get_dummies(train_data['Embarked'])['S']

    # title features
    IsMr = train_data['Name'].apply(lambda x: 1 if 'Mr' in x else 0)
    IsMrs = train_data['Name'].apply(lambda x: 1 if 'Mrs' in x else 0)
    Isminor = pd.get_dummies(train_data['Title'])[3]
    Ismiss = pd.get_dummies(train_data['Title'])[2]
    Israre = pd.get_dummies(train_data['Title'])[4]

    # target feature
    y = train_data['Survived']

    X = pd.concat([Isfemale, Isold, Isalone, C1,
                   C2, Age, Fare, Isq, Iss, IsMr, IsMrs,
                   Isminor, Ismiss, Israre, SibSp, Parch], axis=1, ignore_index=True)

    feature_names = ['Isfemale', 'Isold', 'Isalone', 'C1',
                     'C2', 'Age', 'Fare', 'Isq', 'Iss', 'Ismr', 'Ismrs',
                     'Isminor', 'Ismiss', 'Israre', 'SibSp', 'Parch']

    X.columns = feature_names

    return X, y