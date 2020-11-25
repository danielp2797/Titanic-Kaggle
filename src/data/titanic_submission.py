from kaggle.api.kaggle_api_extended import KaggleApi

def submission(data):
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(data, 'API Submission', 'titanic')

