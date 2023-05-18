import os
import pickle

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index import GPTVectorStoreIndex, download_loader

load_dotenv()
os.environ.get('OPENAI_API_KEY')

app = FastAPI()

origins = [
    "https://papaya-bavarois-c4cea1.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authenticate our Google account to discover Google Docs.
def authorize_gdocs():
    google_oauth2_scopes = [
        "https://www.googleapis.com/auth/documents.readonly"
    ]
    cred = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", 'rb') as token:
            cred = pickle.load(token)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", google_oauth2_scopes)
            cred = flow.run_local_server(port=0)
        with open("token.pickle", 'wb') as token:
            pickle.dump(cred, token)

@app.get("/")
async def root(prompt: str):
    authorize_gdocs()
    GoogleDocsReader = download_loader('GoogleDocsReader')
    gdoc_ids = ['1qmG5KtGh7eanbgM_2fHagz11uhJ-USlHiZQs1nlSJxA']
    loader = GoogleDocsReader()
    documents = loader.load_data(document_ids=gdoc_ids)
    index = GPTVectorStoreIndex.from_documents(documents)

    while True:
        # prompt = input('Type your prompt...')
        query_engine = index.as_query_engine()
        response = query_engine.query(prompt)
        # print(response)

        # Get the last token usage
        # last_token_usage = index.llm_predictor.last_token_usage
        # print(f"last_token_usage={last_token_usage}")
        
        return {"message": response}