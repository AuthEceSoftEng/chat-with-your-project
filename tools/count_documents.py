from services.db_engine import count_documents_service


async def execute(
    repo_name: str | None = None,
    collection: str | None = None,
    filter: dict | None = None,
):
    """
    Count documents in a specified collection.
    Args:
        repo_name:
            The name of the repository to analyze.
        collection:
            The target collection to run the count against.
        filter:
            A MongoDB query filter object.
    """
    return await count_documents_service(repo_name, collection, filter)