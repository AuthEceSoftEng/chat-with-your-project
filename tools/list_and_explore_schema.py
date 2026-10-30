from typing import Literal
from services.db_engine import list_and_explore_schema_service


async def execute(
    repo_name: str | None = None,
    collection_name: Literal[
        "projects",
        "issues",
        "issueEvents",
        "issueComments",
        "commits",
        "commitComments",
        "contributors",
        "violations",
        "commitsLite3",
        "stats"
    ] | None = None,
):
    """
    Inspect the database schema.
    Args:
        repo_name:
            The name of the repository to analyze.
        collection_name:
            Optional collection name to inspect.
            If omitted, all available collections are listed.
    """
    return await list_and_explore_schema_service(
        {
            "collection_name": collection_name,
            "repo_name": repo_name
        }
    )