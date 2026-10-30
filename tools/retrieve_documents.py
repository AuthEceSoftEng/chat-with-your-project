from services.db_engine import retrieve_documents_service


async def execute(
    repo_name: str | None = None,
    collection: str | None = None,
    filter: dict | None = None,
    sort: dict | None = None,
    limit: int | None = None,
    projection: dict | None = None,
):
    """
    Retrieve documents from a MongoDB collection and store the result in the
    in-memory buffer. The returned buffer_id is a handle for downstream tools;
    it is not a MongoDB collection name.
    Args:
        repo_name:
            The name of the repository to analyze.
        collection:
            The target collection to run the retrieval against.
        filter:
            A MongoDB query filter object.
        sort:
            A MongoDB sort object.
        limit:
            The maximum number of documents to return.
        projection:
            A MongoDB projection object.
    Returns:
        A lightweight result containing buffer_id. Pass that buffer_id to
        execute_math_calculation or execute_ml_pipeline instead of querying a
        collection named buf_*.
    """
    return await retrieve_documents_service(repo_name, collection, filter, sort, limit, projection)