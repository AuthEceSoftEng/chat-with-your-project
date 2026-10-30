from services.db_engine import aggregate_collection_service

async def execute(
    repo_name: str | None = None,
    collection: str | None = None,
    pipeline: list[dict] | None = None,
):
    """
    Aggregate documents from a MongoDB collection and store the result in the
    in-memory buffer. The returned buffer_id is a handle for downstream tools;
    it is not a MongoDB collection name.
    Args:
        repo_name:
            The name of the repository to analyze.
        collection:
            The target collection to run the aggregation against.
        pipeline:
            A MongoDB aggregation pipeline object.
    Returns:
        A lightweight result containing buffer_id. Pass that buffer_id to
        execute_math_calculation or execute_ml_pipeline instead of querying a
        collection named buf_*.
    """
    return await aggregate_collection_service(repo_name, collection, pipeline)