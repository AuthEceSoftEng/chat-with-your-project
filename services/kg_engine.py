import os
from neo4j import GraphDatabase
from services.buffer import buffer_registry

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")

async def query_knowledge_graph_service(query: str | None = None):
    """
    Executes a raw Cypher query against the Memgraph knowledge graph database,
    stores the results in the universal buffer, and returns a lightweight buffer reference.
    """
    if not query:
        return "Please provide a valid Cypher query."

    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=("", ""))
    try:
        with driver.session() as session:
            result = session.run(query)
            results = [record.data() for record in result]
    except Exception as e:
        return f"Database Error: {str(e)}"
    finally:
        driver.close()

    # Store heavy graph records in the universal buffer
    buffer_id = buffer_registry.put(results)

    return {
        "records_returned_count": len(results),
        "results": results,
        "buffer_id": buffer_id,
        "message": f"Knowledge graph query executed successfully. Retrieved {len(results)} records and stored them in the buffer under ID '{buffer_id}'. Pass this buffer_id to downstream analytics or ML tools."
    }