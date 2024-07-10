import firebase_admin
from firebase_admin import credentials


class realtime_DB:
    def __init__(self, in1):
        super().__init__()  # initiate base class cache
        # instance attributes
        self.in1 = {}  # private but not protected with __


    def initdb(self):
        # Replace 'path/to/your/serviceAccountKey.json' with the actual path to your JSON file
        cred = credentials.Certificate(
        'C:/Users/DELL/Documents/GitHub/ML_Superv_Reg_RandomForest/db17-22f40-firebase-adminsdk-6ko5w-986a994da9.json')
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://db17-22f40-default-rtdb.firebaseio.com'})


def writedb(selfself):
        pass






