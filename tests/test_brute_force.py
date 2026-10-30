import asyncio
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
import sys

from pymongo import MongoClient
from dotenv import load_dotenv

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from langchain_core.messages import SystemMessage, HumanMessage
from model import get_model

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "gddata"

REPOS = [
    "apache_kafka",
    "apache_tomcat",
    "hibernate_hibernate-orm",
    "spring-projects_spring-boot",
    "jenkinsci_github-plugin"
]

QUESTIONS = [
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
    "How many issue events are associated with issue #2?",
    "Which contributor most recently made a commit?",
    "Which issue was closed most recently?",
    "Which contributor has authored at least one commit comment?",
    "What are the top 5 contributors ranked by number of commits?",

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
    "Which contributor accounts for the highest percentage of all commits?",

    "Forecast the number of commits expected in the next month.",
    "Cluster issues into thematic groups based on their titles and descriptions.",
    "Summarize the most discussed technical topics across last 10 issues and comments.",
    "Assign issue #2 to contributor.",
    "Find duplicate or highly similar issues across 50 examples using semantic similarity.",
    "Forecast the number of issue comments the issue #2 will have.",
    "Cluster the last 10 commits in severity (Low, Medium, High) based on files changed.",
    "Estimate the probability issue #2 will be closed within 30 days.",
    "Based on last 10 commits of top 5 contributors, classify them in their specialty.",
    "Based on last 10 open issues, propose 5 sub teams to split workload."
]

def extract_token_count(response) -> int:
    """Extract total tokens from response usage metadata or response metadata."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return response.usage_metadata.get("total_tokens", 0)
    elif hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})
        return token_usage.get("total_tokens", 0)
    return 0

async def invoke_with_retry(model, messages, max_tries=10):
    """Asynchronous helper function to retry model invocation up to max_tries on any exception with exponential backoff."""
    for attempt in range(1, max_tries + 1):
        try:
            return model.invoke(messages)
        except Exception as e:
            print(f"\n[!] Model invocation error (Attempt {attempt}/{max_tries}): {e}")
            if attempt == max_tries:
                print("[!] Max retries reached. Raising exception.")
                raise e
            wait_time = min(2 ** (attempt - 1), 60)
            print(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

def make_json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

async def generate_mongo_query(aimodel, repo_name: str, question: str) -> str:
    """Ask model to generate a safe Python/PyMongo query snippet for the gddata database."""
    query_system_prompt = (
        f"You are a MongoDB query generator for the repository '{repo_name}'.\n"
        "Generate Python code using only the existing synchronous pymongo database variable `db`. Be concise and factual, and never invent data.\n\n"
        f"CRITICAL RULE: Every query must filter by `repo_name: '{repo_name}'` (use the exact literal string value, do NOT use curly braces around it).\n\n"
        "Collections available: projects, issues, issueEvents, issueComments, commits, commitComments, contributors, violations, commitsLite3, stats.\n\n"
        "RULES FOR CODE GENERATION:\n"
        "- Return ONLY valid Python code inside a ```python markdown block.\n"
        "- NO imports, NO loops, NO print(), NO explanations.\n"
        "- Assign the final result to a variable named `result`.\n"
        "- Keep limits small (e.g., 5-20) when fetching documents.\n"
        "- Never repeat a query with the same logic.\n"
        "- If data is missing, assign an empty list or error message to `result`.\n\n"
        "GOOD EXAMPLE:\n"
        f"result = list(db.commits.find({{'repo_name': '{repo_name}'}}, {{'commit.message': 1, '_id': 0}}).limit(10))"
    )

    prompt = [
        SystemMessage(content=query_system_prompt),
        HumanMessage(content=question)
    ]
    
    response = await invoke_with_retry(aimodel, prompt, max_tries=5)
    text = response.content
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

async def run_agent_batch(db, repo_name: str, questions: list, output_csv: str):
    _, aimodel = get_model()

    answer_system_prompt = (
        f"You are a repository assistant for the repository '{repo_name}'. "
        "Answer questions using only the provided database facts. Be concise and factual, and never invent data.\n\n"
        f"CRITICAL RULE: Ensure all answers strictly correspond to repository '{repo_name}'.\n\n"
        "RESPONSE STYLE:\n"
        "Answer first, then add a single line detailing the filters, projections, and limits used based on the retrieved context. "
        "Label estimates or ML/buffer outputs clearly."
    )

    print(f"Running resilient pipeline for repository '{repo_name}' across {len(questions)} questions...")
    results = []

    try:
        for idx, question in enumerate(questions, start=1):
            print(f"\n--- Processing Q{idx}: {question} ---")
            
            total_tokens = 0
            db_data = None
            
            try:
                code = await generate_mongo_query(aimodel, repo_name, question)
                print(f"  [Generated Query]: {code}")
                
                local_vars = {}
                exec(code, {"db": db}, local_vars)
                db_data = local_vars.get("result", "No result returned")
                db_data = make_json_safe(db_data)
            except Exception as query_err:
                print(f"  [!] Query generation/execution error: {query_err}")
                db_data = f"Error retrieving data: {str(query_err)}"

            context_message = f"Retrieved Database Data for {repo_name}:\n{json.dumps(db_data, default=str)[:30000]}"
            messages = [
                SystemMessage(content=answer_system_prompt),
                HumanMessage(content=context_message),
                HumanMessage(content=question)
            ]
            
            answer = ""
            try:
                response = await invoke_with_retry(aimodel, messages, max_tries=5)
                total_tokens += extract_token_count(response)
                answer = response.content
                print(f"  [AI Answer]: {answer[:200]}...")
            except Exception as inner_e:
                answer = f"Error processing question: {str(inner_e)}"
                print(f"  [!] {answer}")
            
            results.append({
                "question": question, 
                "answer": answer, 
                "token_count": total_tokens
            })

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user (Ctrl + C). Saving current progress...")

    if results:
        fieldnames = ["question", "answer", "token_count"]
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\n[✓] Results successfully saved to '{output_csv}' ({len(results)}/{len(questions)} questions answered).")
    else:
        print("\nNo results were collected to save.")

async def main():
    output_dir = Path("outputs_bf_gemma")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    try:
        for repo in REPOS:
            output_file = output_dir / f"{repo}_answers.csv"
            print(f"\n==========================================")
            print(f"Processing repository: {repo}")
            print(f"==========================================")
            await run_agent_batch(db, repo, QUESTIONS, str(output_file))
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user. Exiting gracefully.")
    finally:
        client.close()
        print("\nMongoDB connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully.")