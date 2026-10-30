from motor.motor_asyncio import AsyncIOMotorClient
import os
from services.buffer import buffer_registry


async def retrieve_documents_service(
    repo_name: str | None = None,
    collection: str | None = None,
    filter: dict | None = None,
    sort: dict | None = None,
    limit: int | None = None,
    projection: dict | None = None,
):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    filter_query = filter or {}
    filter_query["repo_name"] = repo_name

    cursor = db[collection].find(filter_query, projection)

    if sort:
        for field, order in sort.items():
            cursor = cursor.sort(field, order)

    if limit is not None:
        cursor = cursor.limit(limit)

    results = await cursor.to_list(length=limit or 100)

    for doc in results:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])

    # Store heavy payload in the universal buffer
    buffer_id = buffer_registry.put(results)

    return {
        "collection": collection,
        "results": results,
        "records_returned_count": len(results),
        "buffer_id": buffer_id,
        "message": f"Retrieved {len(results)} records and stored them in the buffer under ID '{buffer_id}'. Pass this buffer_id to downstream analytics or ML tools."
    }


async def count_documents_service(
    repo_name: str | None = None,
    collection: str | None = None,
    filter: dict | None = None,
):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    filter_query = filter or {}
    filter_query["repo_name"] = repo_name

    count = await db[collection].count_documents(filter_query)

    return {
        "collection": collection,
        "count": count
    }


async def aggregate_collection_service(
    repo_name: str | None = None,
    collection: str | None = None,
    pipeline: list[dict] | None = None,
):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata

    if pipeline is None:
        return {"error": "No aggregation pipeline provided."}

    if pipeline and isinstance(pipeline[0], dict) and "$match" in pipeline[0]:
        pipeline[0]["$match"]["repo_name"] = repo_name
    else:
        pipeline.insert(0, {"$match": {"repo_name": repo_name}})

    cursor = db[collection].aggregate(pipeline)
    results = await cursor.to_list(length=None)

    for doc in results:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])

    buffer_id = buffer_registry.put(results)

    return {
        "collection": collection,
        "results": results,
        "records_returned_count": len(results),
        "buffer_id": buffer_id,
        "message": f"Aggregated {len(results)} records and stored them in the buffer under ID '{buffer_id}'. Pass this buffer_id to downstream analytics or ML tools."
    }


async def list_and_explore_schema_service(args: dict = {}):
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    db = client.gddata
    repo_name = args.get("repo_name")
    collection_name = args.get("collection_name")
    
    if not collection_name:
        return {
            "available_collections": [
                "projects", "issues", "issueEvents", "issueComments", 
                "commits", "commitComments", "contributors", "violations", "commitsLite3", "stats"
            ],
            "instruction": "Provide a collection_name to view a sample document and its fields."
        }
        
    sample_doc = await db[collection_name].find_one({"repo_name": repo_name})
    if not sample_doc:
        sample_doc = await db[collection_name].find_one({})
        
    if not sample_doc:
        return {"error": f"Collection '{collection_name}' is empty."}
        
    if "_id" in sample_doc:
        sample_doc["_id"] = str(sample_doc["_id"])
        
    return {
        "collection": collection_name,
        "schema_sample_fields": sample_doc
    }
