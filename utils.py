# recommend_contributors.py

import pandas as pd
import numpy as np
import re
import ast
import torch
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from collections import defaultdict
from tabulate import tabulate

import nltk
nltk.download("stopwords")


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(stemmer.stem(word) for word in text.split() if word not in stop_words)

def extract_clean_label(labels, prefix):
    return next((label[len(prefix):] for label in labels if label.startswith(prefix)), None)

def is_decimal_number(s):
    try:
        return '.' in s and s.replace('.', '', 1).isdigit()
    except:
        return False

def extract_file_patterns(text):
    pattern = r'(?:[\w./-]*/)?([\w.-]+(?:_test)?\.(?:[a-zA-Z0-9_]+))|\B(\.\w[\w.-]*)'
    exclude_prefixes = ['github.com', 'git.k8s.io', 'k8s.io', 'http://', 'https://']
    if pd.isna(text): return []
    matches = re.findall(pattern, text)
    files = [m[0] or m[1] for m in matches if m[0] or m[1]]
    cleaned = []
    for f in files:
        if any(domain in f for domain in exclude_prefixes): continue
        if f.count('/') > 2 or is_decimal_number(f): continue
        cleaned.append(f.split('/')[-1])
    return cleaned


def load_data(new_issue_df, pr_path, train_path, con_path):

    issues_df = new_issue_df
    # issues_df = pd.read_csv(new_issue_df)
    prs_df = pd.read_csv(pr_path).head(20000)
    train_df = pd.read_csv(train_path)
    con_df = pd.read_csv(con_path)

    # Preprocess issues

    # Fill missing
    text_cols = ['Body', 'referenced_files', 'mentioned_libraries', 'error_messages', 'api_calls', 'execution_details', 'detected_languages']
    issues_df[text_cols] = issues_df[text_cols].fillna("")
    issues_df['Closed by'] = issues_df['Closed by'].fillna("Unknown")
    issues_df['Labels'] = issues_df['Labels'].fillna("Unknown")

    for col in ["Created At", "Updated At", "Closed At"]:
        issues_df[col] = pd.to_datetime(issues_df[col], errors='coerce')

    for col in ["Labels", "Assignees"]:
        issues_df[col] = issues_df[col].apply(lambda x: x.split(", ") if isinstance(x, str) and x else [])

    issues_df["Labels"] = issues_df["Labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    issues_df["sig_label"] = issues_df["Labels"].apply(lambda x: extract_clean_label(x, 'sig/'))
    issues_df["area_label"] = issues_df["Labels"].apply(lambda x: extract_clean_label(x, 'area/'))
    issues_df["kind_label"] = issues_df["Labels"].apply(lambda x: extract_clean_label(x, 'kind/'))

    issues_df["combined_text"] = (issues_df["Title"].fillna("") + " " + issues_df["Body"].fillna("")).apply(clean_text)
    issues_df["changed_files"] = issues_df["Body"].apply(extract_file_patterns)
    issues_df["Assignees"] = issues_df["Assignees"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    issues_df["changed_files"] = issues_df["changed_files"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    prs_df["Assignees"] = prs_df["Assignees"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Contributor preprocessing
    for col in ['sig_label_stats', 'area_label_stats', 'kind_label_stats', 'changed_files_freq']:
        con_df[col] = con_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return issues_df, prs_df, train_df, con_df


def recommend_contributors_for_issue(issue_row, model, train_vectors, pr_vectors, train_df, prs_df, con_df, ranker_model, scaler, top_k=10):
    test_vec = model.encode([issue_row['combined_text']], convert_to_tensor=True)

    sig_label = issue_row.get('sig_label')
    area_label = issue_row.get('area_label')
    kind_label = issue_row.get('kind_label')
    issue_files = issue_row.get('changed_files', [])

    issue_scores = cos_sim(test_vec, train_vectors)[0]
    top_issue_indices = torch.topk(issue_scores, k=20).indices.tolist()

    final_score_map = defaultdict(lambda: {
        "score": 0.0,
        "similar_issues": [],
        "similar_prs": []
    })

    for idx in top_issue_indices:
        score = issue_scores[idx].item()
        assignees = train_df.iloc[idx]['Assignees']
        assignees = assignees if isinstance(assignees, list) else [assignees]
        for a in assignees:
            final_score_map[a]["score"] += score
            if len(final_score_map[a]["similar_issues"]) < 3:
                final_score_map[a]["similar_issues"].append(train_df.iloc[idx]['Issue number'])

    pr_scores = cos_sim(test_vec, pr_vectors)[0]
    top_pr_indices = torch.topk(pr_scores, k=15).indices.tolist()

    for idx in top_pr_indices:
        row_pr = prs_df.iloc[idx]
        score = pr_scores[idx].item()
        contributors = []
        if isinstance(row_pr['Opened by'], list): contributors.extend(row_pr['Opened by'])
        else: contributors.append(row_pr['Opened by'])

        if isinstance(row_pr['Assignees'], list): contributors.extend(row_pr['Assignees'])
        else: contributors.append(row_pr['Assignees'])

        for c in contributors:
            final_score_map[c]["score"] += score
            if len(final_score_map[c]["similar_prs"]) < 3:
                final_score_map[c]["similar_prs"].append(row_pr['PR number'])

    def get_label_freq(label, stats):
        if not label or not isinstance(stats, list): return 0
        for item in stats:
            if isinstance(item, list) and len(item) == 2 and item[0] == label:
                return item[1]
        return 0

    def get_file_overlap_score(issue_files, changed_files_freq):
        if not isinstance(issue_files, list) or not isinstance(changed_files_freq, list):
            return 0
        return sum(1 for f in issue_files if f in changed_files_freq)

    candidate_rows = []
    for c, info in final_score_map.items():
        con_row = con_df[con_df['user'] == c]
        if con_row.empty: continue
        con_row = con_row.iloc[0]
        candidate_rows.append({
            'contributor': c,
            'score': info['score'],
            'similar_issues': info['similar_issues'],
            'similar_prs': info['similar_prs'],
            'sig_label_score': get_label_freq(sig_label, con_row['sig_label_stats']),
            'area_label_score': get_label_freq(area_label, con_row['area_label_stats']),
            'kind_label_score': get_label_freq(kind_label, con_row['kind_label_stats']),
            'file_overlap_score': get_file_overlap_score(issue_files, con_row['changed_files_freq']),
            'recent_commits': con_row.get('recent_commits', 0),
            'recent_issues': con_row.get('recent_issues', 0),
            'recent_prs': con_row.get('recent_prs', 0)
        })

    if not candidate_rows:
        return [], issue_row['Assignees']

    pred_df = pd.DataFrame(candidate_rows)
    label_features = ['sig_label_score', 'area_label_score', 'kind_label_score', 'recent_commits', 'recent_issues', 'recent_prs']
    features = ['score'] + label_features + ['file_overlap_score']
    pred_df[label_features] = scaler.transform(pred_df[label_features])
    pred_df['rank_score'] = ranker_model.predict(pred_df[features])
    pred_df = pred_df.sort_values(by='rank_score', ascending=False)

    # Build formatted markdown table with rank, contributor, similar issues, similar PRs
    table_data = []
    for rank, (_, row) in enumerate(pred_df.head(top_k).iterrows(), start=1):
        contributor = f"@{row['contributor']}"
        issues = ', '.join([f"#{i}" for i in row['similar_issues']]) or "None"
        prs = ', '.join([f"#{p}" for p in row['similar_prs']]) or "None"
        table_data.append([rank, contributor, issues, prs])

    headers = ["Rank", "Contributor", "Similar Issues", "Similar PRs"]
    markdown_table = tabulate(table_data, headers=headers, tablefmt="github")

    return markdown_table, issue_row['Assignees']


# if __name__ == "__main__":
#     issue_path = 'data/issues_df.csv'
#     pr_path = 'data/prs_df.csv'
#     train_path = 'data/train_df.csv'
#     con_path = 'data/merged_con_availability_stats.csv'

#     issues_df, prs_df, train_df, con_df = load_data(issue_path, pr_path, train_path, con_path)

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     train_vectors = torch.load('models/train_vectors.pt')
#     test_vectors = torch.load('models/test_vectors.pt')
#     pr_vectors = torch.load('models/pr_vectors.pt')
#     scaler = joblib.load('models/label_scaler.pkl')
#     ranker_model = joblib.load('models/ranker_model.pkl')

#     # Example: get recommendation for one issue
#     issue_row = issues_df.iloc[54]
#     recommended, actual = recommend_contributors_for_issue(issue_row, model, train_vectors, pr_vectors, train_df, prs_df, con_df, ranker_model, scaler)
    
#     print("Recommended Contributors:")
#     print(recommended)
#     # print("Actual Assignees:", actual)
