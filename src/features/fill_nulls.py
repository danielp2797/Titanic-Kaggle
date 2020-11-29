import pandas as pd
import numpy as np

def fill_age_values(data):  # titanic train or test data


    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].apply(lambda x:
                                            1 if x in ['Mr', 'Mrs'] else
                                            (2 if x == 'Miss' else
                                             (3 if x == 'Master' else 4)))
    group_means = []

    for age_group in data['Title'].unique().tolist():
        group_means.append(data[data['Title'].isin([age_group])]['Age'].mean())

    means_dict = dict(zip(data['Title'].unique().tolist(), group_means))  # dict to use with replace

    ages_to_fill = data[data['Age'].isnull()]['Title'].replace(means_dict)
    ages_to_fill = ages_to_fill.apply(int)  # transform age into int type
    data.loc[ages_to_fill.index, 'Age'] = ages_to_fill

    print('Age null values filled successfully')
    return

def fill_fare_value(data):  # use with test data

    assert data['Fare'].isnull().sum()>0, 'no null value found'
    mean_fare_of_class = data[data['Pclass'] == 3]['Fare'].mean()
    null_index = data[data['Fare'].isnull()].index[0]
    data.loc[null_index, 'Fare'] = mean_fare_of_class

    data['Fare'] = data['Fare'].astype(float)

    print('Fare value filled successfully')

def fill_embarked(data):  # use with train and test if it neccesary

    if data['Embarked'].isnull().sum()>0:
        mode_embarked = data['Embarked'].mode()[0]
        data.loc[data['Embarked'].isnull(), 'Embarked'] = mode_embarked
    else:
        print('No null values found')

    print('Embarked value filled successfully ')





