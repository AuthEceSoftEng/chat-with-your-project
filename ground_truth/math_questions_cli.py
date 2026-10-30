import os
from motor.motor_asyncio import AsyncIOMotorClient
import argparse
import pandas as pd
import asyncio
import dotenv
from datetime import datetime
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client.gddata
dotenv.load_dotenv()


questions = [
    "What is the average number of commits per contributor?",
    "What percentage of issues are currently closed?",
    "What percentage of issues are currently open?",
    "What is the ratio of open issues to closed issues?",
    "What is the average number of comments per issue?",
    "What is the median number of comments per issue?",
    "What is the average number of issue events per issue?",
    "What is the average number of commit comments per commit?",
    "What percentage of contributors have made more than 10 commits?",
    "What is the standard deviation of commits across contributors?",
    "What is the average time (in days) taken to close an issue?",
    "What is the median time (in days) taken to close an issue?",
    "What percentage of issues were closed within 7 days of creation?",
    "What percentage of issues were closed within 30 days of creation?",
    "Which contributor has the highest commit-to-comment ratio?",
    "What percentage of all comments are issue comments versus commit comments?",
    "What is the average number of commits per month since the repository was created?",
    "What was the percentage growth or decline in commits from the previous month?",
    "What percentage of issues have at least one comment?",
    "Which contributor accounts for the highest percentage of all commits?"
]

async def average_commits_per_contributor(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {"$match": {"repo_name": repo_name}},
        {"$group": {
            "_id": None,
            "average_commits": {"$avg": "$contributions"} 
        }}
    ]
    
    result = await db.contributors.aggregate(pipeline).to_list(length=1)
    return result[0]["average_commits"] if result else 0

async def percentage_closed_issues(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_issues = await db.issues.count_documents({"repo_name": repo_name})
    closed_issues = await db.issues.count_documents({"repo_name": repo_name, "state": "closed"})
    
    if total_issues == 0:
        return 0
    
    return (closed_issues / total_issues) * 100

async def percentage_open_issues(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_issues = await db.issues.count_documents({"repo_name": repo_name})
    open_issues = await db.issues.count_documents({"repo_name": repo_name, "state": "open"})
    
    if total_issues == 0:
        return 0
    
    return (open_issues / total_issues) * 100

async def ratio_open_to_closed_issues(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    closed_issues = await db.issues.count_documents({"repo_name": repo_name, "state": "closed"})
    open_issues = await db.issues.count_documents({"repo_name": repo_name, "state": "open"})
    
    if closed_issues == 0:
        return float('inf') if open_issues > 0 else 0
    
    return open_issues / closed_issues

async def average_comments_per_issue(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {"$match": {"repo_name": repo_name}},
        {"$group": {
            "_id": None,
            "average_comments": {"$avg": "$comments"} 
        }}
    ]
    
    result = await db.issues.aggregate(pipeline).to_list(length=1)
    return result[0]["average_comments"] if result else 0

async def median_comments_per_issue(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    comments_counts = await db.issues.find({"repo_name": repo_name}, {"comments": 1, "_id": 0}).to_list(length=None)
    counts = [doc["comments"] for doc in comments_counts]
    
    if not counts:
        return 0
    
    counts.sort()
    n = len(counts)
    mid = n // 2
    
    if n % 2 == 0:
        return (counts[mid - 1] + counts[mid]) / 2
    else:
        return counts[mid]
    
async def average_issue_events_per_issue(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_issues = await db.issues.count_documents({"repo_name": repo_name})
    total_issue_events = await db.issueEvents.count_documents({"repo_name": repo_name})
    
    if total_issues == 0:
        return 0
    
    return total_issue_events / total_issues

async def average_commit_comments_per_commit(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_commits = await db.commits.count_documents({"repo_name": repo_name})
    total_commit_comments = await db.commitComments.count_documents({"repo_name": repo_name})
    
    if total_commits == 0:
        return 0
    
    return total_commit_comments / total_commits

async def percentage_contributors_more_than_10_commits(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_contributors = await db.contributors.count_documents({"repo_name": repo_name})
    contributors_with_more_than_10_commits = await db.contributors.count_documents({
        "repo_name": repo_name,
        "contributions": {"$gt": 10}
    })
    
    if total_contributors == 0:
        return 0
    
    return (contributors_with_more_than_10_commits / total_contributors) * 100

async def stddev_commits_across_contributors(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    contributions = await db.contributors.find({"repo_name": repo_name}, {"contributions": 1, "_id": 0}).to_list(length=None)
    counts = [doc["contributions"] for doc in contributions]
    stddev = pd.Series(counts).std() if counts else 0
    return stddev

async def average_time_to_close_issue(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    pipeline = [
        {"$match": {"repo_name": repo_name, "state": "closed"}},
        {"$project": {
            "time_to_close": {
                "$divide": [
                    {
                        "$subtract": [
                            {"$dateFromString": {"dateString": "$closed_at"}},
                            {"$dateFromString": {"dateString": "$created_at"}}
                        ]
                    },
                    1000 * 60 * 60 * 24 #ms to days
                ]
            }
        }},
        {"$group": {
            "_id": None,
            "average_time_to_close": {"$avg": "$time_to_close"}
        }}
    ]    
    result = await db.issues.aggregate(pipeline).to_list(length=1)
    return result[0]["average_time_to_close"] if result else 0

async def median_time_to_close_issue(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    closed_issues = await db.issues.find({"repo_name": repo_name, "state": "closed"}, {"created_at": 1, "closed_at": 1, "_id": 0}).to_list(length=None)
    
    time_to_close_days = [
        (datetime.strptime(doc["closed_at"], "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(doc["created_at"], "%Y-%m-%dT%H:%M:%SZ")).total_seconds() / (60 * 60 * 24)
        for doc in closed_issues
    ]
    
    if not time_to_close_days:
        return 0
    
    time_to_close_days.sort()
    n = len(time_to_close_days)
    mid = n // 2
    
    if n % 2 == 0:
        return (time_to_close_days[mid - 1] + time_to_close_days[mid]) / 2
    else:
        return time_to_close_days[mid]
    
async def percentage_issues_closed_within_7_days(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_closed_issues = await db.issues.count_documents({"repo_name": repo_name, "state": "closed"})
    closed_within_7_days = await db.issues.count_documents({
        "repo_name": repo_name,
        "state": "closed",
        "$expr": {
            "$lte": [
                {"$divide": [{"$subtract": [
                            {"$dateFromString": {"dateString": "$closed_at"}},
                            {"$dateFromString": {"dateString": "$created_at"}}
                        ]}, 1000 * 60 * 60 * 24]},
                7
            ]
        }
    })
    
    if total_closed_issues == 0:
        return 0
    
    return (closed_within_7_days / total_closed_issues) * 100

async def percentage_issues_closed_within_30_days(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_closed_issues = await db.issues.count_documents({"repo_name": repo_name, "state": "closed"})
    closed_within_30_days = await db.issues.count_documents({
        "repo_name": repo_name,
        "state": "closed",
        "$expr": {
            "$lte": [
                {"$divide": [{"$subtract": [
                            {"$dateFromString": {"dateString": "$closed_at"}},
                            {"$dateFromString": {"dateString": "$created_at"}}
                        ]}, 1000 * 60 * 60 * 24]},
                30
            ]
        }
    })
    
    if total_closed_issues == 0:
        return 0
    
    return (closed_within_30_days / total_closed_issues) * 100

async def contributor_highest_commit_to_comment_ratio(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    contributors = await db.contributors.find({"repo_name": repo_name}).to_list(length=None)
    
    highest_ratio = 0
    top_contributor = None
    
    for contributor in contributors:
        commits = contributor.get("contributions", 0)
        comments = contributor.get("comments", 0)
        
        if comments == 0:
            ratio = float('inf') if commits > 0 else 0
        else:
            ratio = commits / comments
            
        if ratio > highest_ratio:
            highest_ratio = ratio
            top_contributor = contributor.get("login") or contributor.get("author")
    
    return top_contributor

async def percentage_issue_comments_vs_commit_comments(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_issue_comments = await db.issueComments.count_documents({"repo_name": repo_name})
    total_commit_comments = await db.commitComments.count_documents({"repo_name": repo_name})
    
    total_comments = total_issue_comments + total_commit_comments
    
    if total_comments == 0:
        return 0
    
    return (total_issue_comments / total_comments) * 100

async def average_commits_per_month(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    project_doc = await db.projects.find_one({"repo_name": repo_name})
    
    if not project_doc:
        return 0
    
    created_at = project_doc.get("created_at")
    if not created_at:
        return 0
    
    total_commits = await db.commits.count_documents({"repo_name": repo_name})
    
    created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    current_date = datetime.now()
    
    months_active = (current_date.year - created_date.year) * 12 + (current_date.month - created_date.month)
    
    if months_active <= 0:
        return total_commits
    
    return total_commits / months_active

async def percentage_growth_decline_commits_previous_month(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    current_date = datetime.now()
    first_day_current_month = datetime(current_date.year, current_date.month, 1)
    first_day_previous_month = datetime(current_date.year, current_date.month - 1, 1) if current_date.month > 1 else datetime(current_date.year - 1, 12, 1)
    
    commits_current_month = await db.commits.count_documents({
        "repo_name": repo_name,
        "date": {"$gte": first_day_current_month.isoformat()}
    })
    
    commits_previous_month = await db.commits.count_documents({
        "repo_name": repo_name,
        "date": {"$gte": first_day_previous_month.isoformat(), "$lt": first_day_current_month.isoformat()}
    })
    
    if commits_previous_month == 0:
        return float('inf') if commits_current_month > 0 else 0
    
    return ((commits_current_month - commits_previous_month) / commits_previous_month) * 100

async def percentage_issues_with_at_least_one_comment(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_issues = await db.issues.count_documents({"repo_name": repo_name})
    issues_with_comments = await db.issues.count_documents({
        "repo_name": repo_name,
        "comments": {"$gt": 0}
    })
    
    if total_issues == 0:
        return 0
    
    return (issues_with_comments / total_issues) * 100

async def contributor_highest_percentage_of_all_commits(repo_name: str):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    total_commits = await db.commits.count_documents({"repo_name": repo_name})
    
    if total_commits == 0:
        return None
    
    pipeline = [
        {"$match": {"repo_name": repo_name}},
        {"$group": {
            "_id": "$author",
            "commit_count": {"$sum": 1}
        }},
        {"$sort": {"commit_count": -1}},
        {"$limit": 1}
    ]
    
    top_contributor = await db.commits.aggregate(pipeline).to_list(length=1)
    
    if not top_contributor:
        return None
    
    top_contributor_data = top_contributor[0]
    percentage_of_all_commits = (top_contributor_data["commit_count"] / total_commits) * 100
    
    author_id = top_contributor_data.get("_id")
    
    # Safely extract the contributor login/name whether _id is a dict, string, or None
    if isinstance(author_id, dict):
        contributor_name = author_id.get("login") or author_id.get("name")
    else:
        contributor_name = author_id
    
    return {
        "contributor": contributor_name,
        "percentage_of_all_commits": percentage_of_all_commits
    }

async def gather_results(repo_name: str):
    return [
        (questions[0], await average_commits_per_contributor(repo_name)),
        (questions[1], await percentage_closed_issues(repo_name)),
        (questions[2], await percentage_open_issues(repo_name)),
        (questions[3], await ratio_open_to_closed_issues(repo_name)),
        (questions[4], await average_comments_per_issue(repo_name)),
        (questions[5], await median_comments_per_issue(repo_name)),
        (questions[6], await average_issue_events_per_issue(repo_name)),
        (questions[7], await average_commit_comments_per_commit(repo_name)),
        (questions[8], await percentage_contributors_more_than_10_commits(repo_name)),
        (questions[9], await stddev_commits_across_contributors(repo_name)),
        (questions[10], await average_time_to_close_issue(repo_name)),
        (questions[11], await median_time_to_close_issue(repo_name)),
        (questions[12], await percentage_issues_closed_within_7_days(repo_name)),
        (questions[13], await percentage_issues_closed_within_30_days(repo_name)),
        (questions[14], await contributor_highest_commit_to_comment_ratio(repo_name)),
        (questions[15], await percentage_issue_comments_vs_commit_comments(repo_name)),
        (questions[16], await average_commits_per_month(repo_name)),
        (questions[17], await percentage_growth_decline_commits_previous_month(repo_name)),
        (questions[18], await percentage_issues_with_at_least_one_comment(repo_name)),
        (questions[19], await contributor_highest_percentage_of_all_commits(repo_name))
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_name", help="Name of the repository")
    args = parser.parse_args()
    print(f"Gathering results for repository: {args.repo_name}")
    results_data = asyncio.run(gather_results(args.repo_name))
    df = pd.DataFrame(results_data, columns=['Question', 'Result'])
    out = f"{args.repo_name}_math_results.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()