from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
from github import Github
from utils import recommend_contributors_for_issue, load_data
import pandas as pd
import torch
import joblib
from sentence_transformers import SentenceTransformer
import jwt
import time
import requests
import os

# Load environment variables
# load_dotenv()
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# REPO_NAME = os.getenv("REPO_NAME")

# # GitHub setup
# app = FastAPI()
# gh = Github(GITHUB_TOKEN)
# repo = gh.get_repo(REPO_NAME)
import os
import time
import requests
import jwt
from fastapi import FastAPI
from dotenv import load_dotenv
import torch
import joblib
from sentence_transformers import SentenceTransformer

import os
import time
import requests
import jwt
from fastapi import FastAPI
from dotenv import load_dotenv
import torch
import joblib
from sentence_transformers import SentenceTransformer
from github import Github  # Needed to use Github API

# Load environment variables
load_dotenv()
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
INSTALLATION_ID = os.getenv("INSTALLATION_ID")
REPO_NAME = os.getenv("REPO_NAME")
PRIVATE_KEY_PATH = "private-key.pem"

# Read private key
with open(PRIVATE_KEY_PATH, "r") as key_file:
    private_key = key_file.read()

# Create JWT
payload = {
    "iat": int(time.time()) - 60,
    "exp": int(time.time()) + (9 * 60),
    "iss": GITHUB_APP_ID
}
jwt_token = jwt.encode(payload, private_key, algorithm="RS256")

# Exchange JWT for installation access token
headers = {
    "Authorization": f"Bearer {jwt_token}",
    "Accept": "application/vnd.github+json"
}
url = f"https://api.github.com/app/installations/{INSTALLATION_ID}/access_tokens"
response = requests.post(url, headers=headers)

if response.status_code == 201:
    access_token = response.json()["token"]
    print("Got access token:", access_token[:10], "...")
else:
    print("Failed to get access token:", response.status_code)
    print(response.text)
    raise Exception("Failed to retrieve GitHub App installation token")

# Initialize GitHub client and repository object
gh = Github(access_token)
repo = gh.get_repo(REPO_NAME)

# Load models and data
print("Loading models and data...")
issue_path = 'data/issues_df.csv'
pr_path = 'data/prs_df.csv'
train_path = 'data/train_df.csv'
con_path = 'data/merged_con_availability_stats.csv'

# issues_df, prs_df, train_df, con_df = load_data(issue_path, pr_path, train_path, con_path)
model = SentenceTransformer('all-MiniLM-L6-v2')
train_vectors = torch.load('models/train_vectors.pt')
pr_vectors = torch.load('models/pr_vectors.pt')
ranker_model = joblib.load('models/ranker_model.pkl')
scaler = joblib.load('models/label_scaler.pkl')

print("Models and data loaded.")

# FastAPI app
app = FastAPI()



@app.post("/webhook")
async def github_webhook(req: Request):
    event = req.headers.get('x-github-event')
    payload = await req.json()

    if event == 'issues':
        action = payload.get('action')
        issue = payload.get('issue')
        issue_number = issue['number']

        if action == 'opened':
            repo.get_issue(issue_number).create_comment("Need help assigning this issue? Type `/recommend` below!")

    elif event == 'issue_comment':
        comment_body = payload['comment']['body']
        if comment_body.strip() == "/recommend":
            issue = payload['issue']
            issue_number = issue['number']

            # Extract title, body and basic GitHub info
            new_issue_data = {
                'Issue number': issue_number,
                'Title': issue['title'],
                'Body': issue['body'] or '',
                'Labels': [label['name'] for label in issue.get('labels', [])],
                'Assignees': [assignee['login'] for assignee in issue.get('assignees', [])],
                'Created At': issue['created_at'],
                'Updated At': issue['updated_at'],
                'Closed At': issue.get('closed_at'),
                'Closed by': issue.get('closed_by', {}).get('login') if issue.get('closed_by') else 'Unknown',
                'referenced_files': '',
                'mentioned_libraries': '',
                'error_messages': '',
                'api_calls': '',
                'execution_details': '',
                'detected_languages': ''
            }

            # Convert to single-row DataFrame and apply your pipeline
            new_issue_df = pd.DataFrame([new_issue_data])
            # Preprocess using same logic as training data (reuse  utils pipeline)
            # Make sure load_data() or a new helper applies all processing (e.g., combined_text, label parsing)
            new_issue_df, prs_df, train_df, con_df = load_data(new_issue_df, pr_path, train_path, con_path)

            # Recommend contributors
            recommended, actual = recommend_contributors_for_issue(
                issue_row=new_issue_df.iloc[0],
                model=model,
                train_vectors=train_vectors,
                pr_vectors=pr_vectors,
                train_df=train_df,
                prs_df=prs_df,
                con_df=con_df,
                ranker_model=ranker_model,
                scaler=scaler
            )

            # Prepare response comment
            if recommended:
                response = "Recommended contributors for this issue:\n\n" + recommended
                # for i, contributor in enumerate(recommended, 1):
                #     response += f"{i}. @{contributor}\n"
            else:
                response = "Sorry, I couldn't find a suitable contributor at this time."

            # Post comment to GitHub issue
            repo.get_issue(issue_number).create_comment(response)

    return {"message": "Event processed"}
