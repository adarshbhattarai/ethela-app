import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from app.api.user_routes import thela_routes
import app.start


app = FastAPI()

def setup_cors():
    origins = [
        "*",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def initialize_firebase():
    if not firebase_admin._apps:
        service_account_key = "E:/codes/GeminiAgent/thela-gcloud-run-apis/service-account.json"
        cred = credentials.Certificate(service_account_key)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db

def setup_routes():
    app.include_router(thela_routes.router, prefix="/users")

setup_cors()
setup_routes()
#db = initialize_firebase()
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
