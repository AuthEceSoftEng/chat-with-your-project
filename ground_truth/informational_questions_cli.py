import os
from motor.motor_asyncio import AsyncIOMotorClient
import argparse
import pandas as pd
import asyncio
import dotenv
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client.gddata
dotenv.load_dotenv()
questions = [
    "What is the name and description of the repository?",
    "When was the repository created, and when was it last updated?",
    "How many contributors have contributed to this repository?",
    "Which contributor has made the highest number of commits?",
    "How many total commits are recorded in the repository?",
    "What is the most recent commit message and who authored it?",
    "What is the oldest commit in the repository?",
    "How many open issues are currently in the repository?",
    "How many closed issues are currently in the repository?",
    "Which issue was created most recently?",
    "Which issue has been open for the longest time?",
    "Which issue has received the highest number of comments?",
    "How many issue comments exist in total?",
    "How many commit comments exist in total?",
    "Which project contains the largest number of issues?",
    "How many issue events are associated with issue #X?",
    "Which contributor most recently made a commit?",
    "Which issue was closed most recently?",
    "Which contributor has authored at least one commit comment?",
    "What are the top 5 contributors ranked by number of commits?"
]

async def name_description(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    repo_doc = await db.projects.find_one({"repo_name": repo_name})
    name = repo_doc.get("name") or ""
    desc = repo_doc.get("description") or ""
    return f"Name: {name}; Description: {desc}"

async def creation_update_dates(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    repo_doc = await db.projects.find_one({"repo_name": repo_name})
    created = repo_doc.get("created_at") or ""
    updated = repo_doc.get("updated_at") or ""
    return f"Created: {created}; Updated: {updated}"

async def total_contributors(repo_name: str) -> str:
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    
    pipeline = [
        {"$match": {"repo_name": repo_name}},
        {"$group": {"_id": "$login"}},
        {"$count": "total"}
    ]
    
    res = await db.contributors.aggregate(pipeline).to_list(length=1)
    return str(res[0]["total"]) if res else "0"

async def highest_commit_contributor(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {"$match": {"repo_name": repo_name}},
        {
            "$addFields": {
                "author_name": {
                    "$ifNull": ["$commit.author.name", "Unknown"]
                }
            }
        },
        {"$group": {"_id": "$author_name", "commit_count": {"$sum": 1}}},
        {"$sort": {"commit_count": -1}},
        {"$limit": 1}
    ]
    cur = db.commits.aggregate(pipeline)
    res = await cur.to_list(length=1)
    if not res:
        return "No commits found"
    top = res[0]
    author = top.get("_id") or "Unknown"
    count = top.get("commit_count", 0)
    return f"{author} ({count} commits)"

async def total_commits(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    count = await db.commits.count_documents({"repo_name": repo_name})
    return str(count)

async def most_recent_commit(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata

    doc = await db.commits.find_one(
        {"repo_name": repo_name},
        sort=[("commit.author.date", -1)]
    )
    if not doc:
        return "No commits found"
    message = doc.get("commit", {}).get("message", "Unknown")
    author = (
        doc.get("author", {}).get("login")
        if doc.get("author")
        else doc.get("commit", {}).get("author", {}).get("name", "Unknown")
    )
    date = doc.get("commit", {}).get("author", {}).get("date", "Unknown")
    return f"Date: {date} Message: {message}; Author: {author}"


async def oldest_commit(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata

    doc = await db.commits.find_one(
        {"repo_name": repo_name},
        sort=[("commit.author.date", 1)]
    )
    if not doc:
        return "No commits found"
    message = doc.get("commit", {}).get("message", "Unknown")
    author = (
        doc.get("author", {}).get("login")
        if doc.get("author")
        else doc.get("commit", {}).get("author", {}).get("name", "Unknown")
    )
    date = doc.get("commit", {}).get("author", {}).get("date", "Unknown")
    return f"Date: {date} Message: {message}; Author: {author}"
    

async def total_open_issues(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    count = await db.issues.count_documents({"repo_name": repo_name, "state": "open"})
    return str(count)

async def total_closed_issues(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    count = await db.issues.count_documents({"repo_name": repo_name, "state": "closed"})
    return str(count)

async def most_recent_created_issue(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    doc = await db.issues.find_one(
        {"repo_name": repo_name},
        sort=[("created_at", -1)]
    )
    if not doc:
        return "No issues found"
    title = doc.get("title", "Unknown")
    created_at = doc.get("created_at", "Unknown")
    return f"Title: {title}; Created at: {created_at}"

async def longest_open_issue(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    doc = await db.issues.find_one(
        {"repo_name": repo_name, "state": "open"},
        sort=[("created_at", 1)]
    )
    if not doc:
        return "No open issues found"
    title = doc.get("title", "Unknown")
    created_at = doc.get("created_at", "Unknown")
    return f"Title: {title}; Created at: {created_at}"

async def highest_commented_issue(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {
            "$match": {
                "repo_name": repo_name,
                "issue_url": {"$type": "string", "$ne": ""}
            }
        },
        {
            "$addFields": {
                "regex_match": {
                    "$arrayElemAt": [
                        {"$regexFindAll": {"input": "$issue_url", "regex": r"\d+$"}},
                        0
                    ]
                }
            }
        },
        {
            "$addFields": {
                "parsed_issue_number": {
                    "$convert": {
                        "input": "$regex_match.match",
                        "to": "int",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        },
        {"$match": {"parsed_issue_number": {"$ne": None}}},
        {"$group": {"_id": "$parsed_issue_number", "comments": {"$sum": 1}}},
        {"$sort": {"comments": -1}},
        {"$limit": 1}
    ]
    cur = db.issueComments.aggregate(pipeline)
    res = await cur.to_list(length=1)
    if not res:
        return "No issue comments found"
    top = res[0]
    issue_number = top.get("_id")
    count = top.get("comments", 0)
    issue_doc = await db.issues.find_one({"repo_name": repo_name, "number": issue_number})
    title = issue_doc.get("title", "Unknown") if issue_doc else "Unknown"
    return f"Issue #{issue_number} ({title}) with {count} comments"

async def total_issue_comments(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    count = await db.issueComments.count_documents({"repo_name": repo_name})
    return str(count)

async def total_commit_comments(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    count = await db.commitComments.count_documents({"repo_name": repo_name})
    return str(count)

async def issue_events_count(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    # Hardcoded issue number 2
    count = await db.issueEvents.count_documents({"repo_name": repo_name, "issue.number": 2})
    return str(count)

async def most_recent_commit_contributor(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    doc = await db.commits.find_one(
        {"repo_name": repo_name},
        sort=[("commit.author.date", -1)]
    )
    if not doc:
        return "No commits found"
    author = (
        doc.get("author", {}).get("login")
        if doc.get("author")
        else doc.get("commit", {}).get("author", {}).get("name", "Unknown")
    )
    date = doc.get("commit", {}).get("author", {}).get("date", "Unknown")
    return f"Author: {author}; Date: {date}"

async def most_recently_closed_issue(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    doc = await db.issues.find_one(
        {"repo_name": repo_name, "state": "closed"},
        sort=[("closed_at", -1)]
    )
    if not doc:
        return "No closed issues found"
    title = doc.get("title", "Unknown")
    closed_at = doc.get("closed_at", "Unknown")
    return f"Title: {title}; Closed at: {closed_at}"

async def contributor_with_commit_comment(repo_name):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    doc = await db.commitComments.find_one({"repo_name": repo_name})
    if not doc:
        return "No commit comments found"
    author = (
        doc.get("user", {}).get("login")
        if doc.get("user")
        else "Unknown"
    )
    return f"Contributor: {author}"

async def top_5_contributors_by_commits(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {"$match": {"repo_name": repo_name}},
        {
            "$addFields": {
                "author_name": {
                    "$ifNull": ["$author.login", "$commit.author.name"]
                }
            }
        },
        {"$match": {"author_name": {"$ne": None}}},
        {"$group": {"_id": "$author_name", "commit_count": {"$sum": 1}}},
        {"$sort": {"commit_count": -1}},
        {"$limit": 5}
    ]
    cur = db.commits.aggregate(pipeline)
    res = await cur.to_list(length=5)
    if not res:
        return "No commits found"        
    top_contributors = []
    for contributor in res:
        author = contributor.get("_id") or "Unknown"
        count = contributor.get("commit_count", 0)
        top_contributors.append(f"{author} ({count} commits)")
        
    return "; ".join(top_contributors)

async def project_with_most_issues():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {"$group": {"_id": "$repo_name", "issue_count": {"$sum": 1}}},
        {"$sort": {"issue_count": -1}},
        {"$limit": 1}
    ]
    cur = db.issues.aggregate(pipeline)
    res = await cur.to_list(length=1)
    if not res:
        return "No issues found"
    top = res[0]
    repo_name = top.get("_id")
    count = top.get("issue_count", 0)
    return f"Repository: {repo_name} with {count} issues"

async def gather_results(repo_name):
    return [
        (questions[0], await name_description(repo_name)),
        (questions[1], await creation_update_dates(repo_name)),
        (questions[2], await total_contributors(repo_name)),
        (questions[3], await highest_commit_contributor(repo_name)),
        (questions[4], await total_commits(repo_name)),
        (questions[5], await most_recent_commit(repo_name)),
        (questions[6], await oldest_commit(repo_name)),
        (questions[7], await total_open_issues(repo_name)),
        (questions[8], await total_closed_issues(repo_name)),
        (questions[9], await most_recent_created_issue(repo_name)),
        (questions[10], await longest_open_issue(repo_name)),
        (questions[11], await highest_commented_issue(repo_name)),
        (questions[12], await total_issue_comments(repo_name)),
        (questions[13], await total_commit_comments(repo_name)),
        (questions[14], await project_with_most_issues()),
        (questions[15], await issue_events_count(repo_name)), #issue 2 hardcoded
        (questions[16], await most_recent_commit_contributor(repo_name)),
        (questions[17], await most_recently_closed_issue(repo_name)),
        (questions[18], await contributor_with_commit_comment(repo_name)),
        (questions[19], await top_5_contributors_by_commits(repo_name))
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_name", help="Name of the repository")
    args = parser.parse_args()
    print(f"Gathering results for repository: {args.repo_name}")
    results_data = asyncio.run(gather_results(args.repo_name))
    df = pd.DataFrame(results_data, columns=['Question', 'Result'])
    out = f"{args.repo_name}_info_results.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()