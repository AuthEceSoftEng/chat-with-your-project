from pymongo import MongoClient
from neo4j import GraphDatabase
import os
import re
from dotenv import load_dotenv

TARGET_REPO = "<REPO_NAME>"
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "gddata"
MEMGRAPH_URI = "bolt://localhost:7687"

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
memgraph_driver = GraphDatabase.driver(MEMGRAPH_URI, auth=("", ""))

BATCH_SIZE = 1000

def run_cypher(query, params=None):
    with memgraph_driver.session() as session:
        session.run(query, params or {})

def create_indexes():
    indexes = [
        "CREATE INDEX ON :Repository(name);",
        "CREATE INDEX ON :Commit(sha);",
        "CREATE INDEX ON :Contributor(login);",
        "CREATE INDEX ON :Issue(number);",
        "CREATE INDEX ON :IssueComment(id);",
        "CREATE INDEX ON :IssueEvent(id);",
        "CREATE INDEX ON :CommitComment(id);",
        "CREATE INDEX ON :File(path);"
    ]
    for idx_query in indexes:
        try:
            run_cypher(idx_query)
        except Exception:
            pass

def resolve_contributor_identity(doc):
    """
    Deterministically resolves contributor identity to match Mongo exactly.
    1. GitHub login (if linked)
    2. Git email (if unlinked CLI commit)
    3. Git author name (if email missing)
    4. None (if completely empty)
    """
    for field in ["author", "user", "actor"]:
        obj = doc.get(field)
        if isinstance(obj, dict) and obj.get("login"):
            return str(obj.get("login")).strip()
            
    if doc.get("author_login"):
        return str(doc.get("author_login")).strip()

    commit_author = doc.get("commit", {}).get("author", {}) if isinstance(doc.get("commit"), dict) else {}
    git_email = commit_author.get("email")
    git_name = commit_author.get("name")

    if git_email:
        return f"git:{git_email.lower().strip()}"
    if git_name:
        return f"git:{git_name.strip()}"

    return None

def extract_issue_number(doc):
    """Parses issue number reliably across all MongoDB collections."""
    issue_obj = doc.get("issue")
    if isinstance(issue_obj, dict) and issue_obj.get("number"):
        try:
            return int(issue_obj["number"])
        except (ValueError, TypeError):
            pass

    issue_url = doc.get("issue_url", "")
    if issue_url:
        match = re.search(r"/issues/(\d+)$", str(issue_url).rstrip("/"))
        if match:
            return int(match.group(1))

    for field in ["number", "issue_number", "issue_id"]:
        val = doc.get(field)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    return None

def migrate_single_repo():
    create_indexes()
    print(f"--- Starting Migration for: '{TARGET_REPO}' ---")

    projects = list(db["projects"].find({"repo_name": TARGET_REPO}))
    stats = list(db["stats"].find({"repo_name": TARGET_REPO}))
    p = projects[0] if projects else {}
    s = stats[0] if stats else {}

    query_repo = """
    MERGE (r:Repository {name: $name})
    SET r.description = $description,
        r.created_at = $created_at,
        r.updated_at = $updated_at,
        r.total_commits = $total_commits,
        r.open_issues = $open_issues
    """
    run_cypher(query_repo, {
        "name": TARGET_REPO,
        "description": p.get("description", ""),
        "created_at": str(p.get("created_at", "")),
        "updated_at": str(p.get("updated_at", "")),
        "total_commits": s.get("total_commits", 0),
        "open_issues": s.get("open_issues", 0)
    })

    print("Migrating Commits & Files...")
    # coll_commits = "commitsLite3" if "commitsLite3" in db.list_collection_names() else "commits"
    coll_commits = "commits"
    cursor = db[coll_commits].find({"repo_name": TARGET_REPO})

    query_commits = """
    UNWIND $batch AS row
    MERGE (c:Commit {sha: row.sha})
    SET c.message = row.message, c.timestamp = row.timestamp

    MERGE (r:Repository {name: row.repo_name})
    MERGE (c)-[:BELONGS_TO]->(r)

    FOREACH (_ IN CASE WHEN row.author IS NOT NULL THEN [1] ELSE [] END |
        MERGE (u:Contributor {login: row.author})
        MERGE (u)-[:AUTHORED]->(c)
    )

    FOREACH (filePath IN row.files |
        MERGE (f:File {path: filePath, repo_name: row.repo_name})
        MERGE (c)-[:MODIFIED]->(f)
        MERGE (f)-[:BELONGS_TO]->(r)
    )
    """
    batch = []
    for doc in cursor:
        sha = str(doc.get("sha") or doc.get("_id"))
        commit_obj = doc.get("commit", {}) if isinstance(doc.get("commit"), dict) else {}
        message = commit_obj.get("message", "")
        timestamp = str(commit_obj.get("author", {}).get("date", ""))
        files_raw = doc.get("files", [])
        file_paths = [f.get("filename") for f in files_raw if isinstance(f, dict) and f.get("filename")]

        batch.append({
            "sha": sha,
            "message": message,
            "timestamp": timestamp,
            "author": resolve_contributor_identity(doc),
            "files": file_paths,
            "repo_name": TARGET_REPO
        })
        if len(batch) >= BATCH_SIZE:
            run_cypher(query_commits, {"batch": batch})
            batch = []
    if batch:
        run_cypher(query_commits, {"batch": batch})

    print("Migrating Issues...")
    cursor = db["issues"].find({"repo_name": TARGET_REPO})

    query_issues = """
    UNWIND $batch AS row
    MERGE (i:Issue {number: row.number, repo_name: row.repo_name})
    SET i.title = row.title,
        i.state = row.state,
        i.priority = row.priority,
        i.created_at = row.created_at,
        i.closed_at = row.closed_at

    MERGE (r:Repository {name: row.repo_name})
    MERGE (i)-[:BELONGS_TO]->(r)

    FOREACH (_ IN CASE WHEN row.author IS NOT NULL THEN [1] ELSE [] END |
        MERGE (u:Contributor {login: row.author})
        MERGE (u)-[:CREATED]->(i)
    )
    """
    batch = []
    for doc in cursor:
        num = extract_issue_number(doc)
        if num is None:
            continue

        batch.append({
            "number": num,
            "title": doc.get("title", ""),
            "state": doc.get("state", "open"),
            "priority": doc.get("priority", "medium"),
            "created_at": str(doc.get("created_at", "")),
            "closed_at": str(doc.get("closed_at", "")),
            "author": resolve_contributor_identity(doc),
            "repo_name": TARGET_REPO
        })
        if len(batch) >= BATCH_SIZE:
            run_cypher(query_issues, {"batch": batch})
            batch = []
    if batch:
        run_cypher(query_issues, {"batch": batch})

    print("Migrating Issue Comments...")
    cursor = db["issueComments"].find({"repo_name": TARGET_REPO})

    query_comments = """
    UNWIND $batch AS row
    MERGE (ic:IssueComment {id: row.id})
    SET ic.body = row.body, ic.created_at = row.created_at

    MERGE (i:Issue {number: row.issue_number, repo_name: row.repo_name})
    MERGE (ic)-[:ON_ISSUE]->(i)

    FOREACH (_ IN CASE WHEN row.author IS NOT NULL THEN [1] ELSE [] END |
        MERGE (u:Contributor {login: row.author})
        MERGE (u)-[:COMMENTED]->(ic)
    )
    """
    batch = []
    for doc in cursor:
        issue_number = extract_issue_number(doc)
        if issue_number is None:
            continue

        comment_id = str(doc.get("_id") or doc.get("id"))
        batch.append({
            "id": comment_id,
            "body": doc.get("body", ""),
            "created_at": str(doc.get("created_at", "")),
            "author": resolve_contributor_identity(doc),
            "issue_number": issue_number,
            "repo_name": TARGET_REPO
        })
        if len(batch) >= BATCH_SIZE:
            run_cypher(query_comments, {"batch": batch})
            batch = []
    if batch:
        run_cypher(query_comments, {"batch": batch})

    print("Migrating Issue Events...")
    cursor = db["issueEvents"].find({"repo_name": TARGET_REPO})

    query_events = """
    UNWIND $batch AS row
    MERGE (ie:IssueEvent {id: row.id})
    SET ie.event = row.event, ie.created_at = row.created_at

    MERGE (i:Issue {number: row.issue_number, repo_name: row.repo_name})
    MERGE (ie)-[:ON_ISSUE]->(i)

    FOREACH (_ IN CASE WHEN row.actor IS NOT NULL THEN [1] ELSE [] END |
        MERGE (u:Contributor {login: row.actor})
        MERGE (u)-[:PERFORMED]->(ie)
    )
    """
    batch = []
    for doc in cursor:
        issue_number = extract_issue_number(doc)
        if issue_number is None:
            continue

        event_id = str(doc.get("_id") or doc.get("id"))
        batch.append({
            "id": event_id,
            "event": doc.get("event", ""),
            "created_at": str(doc.get("created_at", "")),
            "actor": resolve_contributor_identity(doc),
            "issue_number": issue_number,
            "repo_name": TARGET_REPO
        })
        if len(batch) >= BATCH_SIZE:
            run_cypher(query_events, {"batch": batch})
            batch = []
    if batch:
        run_cypher(query_events, {"batch": batch})

    print("Migrating Commit Comments...")
    cursor = db["commitComments"].find({"repo_name": TARGET_REPO})

    query_commit_comments = """
    UNWIND $batch AS row
    MERGE (cc:CommitComment {id: row.id})
    SET cc.body = row.body,
        cc.created_at = row.created_at,
        cc.path = row.path,
        cc.position = row.position

    MERGE (c:Commit {sha: row.commit_sha})
    MERGE (cc)-[:ON_COMMIT]->(c)

    FOREACH (_ IN CASE WHEN row.author IS NOT NULL THEN [1] ELSE [] END |
        MERGE (u:Contributor {login: row.author})
        MERGE (u)-[:COMMENTED]->(cc)
    )
    """
    batch = []
    for doc in cursor:
        commit_sha = doc.get("commit_id") or doc.get("commit_sha") or doc.get("sha")
        if not commit_sha:
            continue

        comment_id = str(doc.get("_id") or doc.get("id"))
        batch.append({
            "id": comment_id,
            "body": doc.get("body", ""),
            "created_at": str(doc.get("created_at", "")),
            "path": doc.get("path", ""),
            "position": doc.get("position"),
            "author": resolve_contributor_identity(doc),
            "commit_sha": str(commit_sha),
            "repo_name": TARGET_REPO
        })
        if len(batch) >= BATCH_SIZE:
            run_cypher(query_commit_comments, {"batch": batch})
            batch = []
    if batch:
        run_cypher(query_commit_comments, {"batch": batch})

    print(f"Successfully finished migration for '{TARGET_REPO}'!")

if __name__ == "__main__":
    migrate_single_repo()