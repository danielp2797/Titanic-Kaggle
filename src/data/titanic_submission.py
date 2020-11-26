from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os

def submit_result(test_prepared, model, features):

    api = KaggleApi()
    api.authenticate()

    id_col = pd.read_csv('../data/raw/test.csv')['PassengerId']

    submission_result = pd.DataFrame(list(zip(id_col, model.predict(test_prepared[features]))),
                                     columns=['PassengerId', 'Survived'])

    submission_result.to_csv('../src/data/submission_result.csv', index=False)
    abs_path = os.path.abspath("../src/data/submission_result.csv")

    api.competition_submit(abs_path, 'API Submission', 'titanic')
