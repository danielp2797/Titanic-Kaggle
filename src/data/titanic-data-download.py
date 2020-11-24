import os
import zipfile

os.system('kaggle competitions download -c "titanic"')

with zipfile.ZipFile('titanic.zip', 'r') as zip_ref:
    zip_ref.extractall('../../data/raw/')

os.remove("titanic.zip")
