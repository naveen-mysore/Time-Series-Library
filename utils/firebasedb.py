import datetime
import dataclasses
import uuid
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

@dataclasses.dataclass
class Metric1:
    mse: float
    mae: float

@dataclasses.dataclass
class Metric2:
    smape: float
    mase: float
    oaw: float

@dataclasses.dataclass
class Task:
    name: str
    dataset: str
    execution_time_minutes: int
    metric1: Metric1
    metric2: Metric2
    seq_len: int
    pred_len: int

@dataclasses.dataclass
class Method:
    name: str
    task: Task

@dataclasses.dataclass
class Result:
    name: str
    machine_name: str
    timestamp: datetime.datetime
    experiment_id: str
    method: Method

class Firebase:
    def __init__(self):
        cred = credentials.Certificate("creds/service_account_key.json")
        self.app = firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def insert(self, collection, doc_id, data):
        doc_ref = self.db.collection(collection).document(doc_id)
        doc_ref.set(data)
