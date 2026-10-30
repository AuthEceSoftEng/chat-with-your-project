from services.kg_engine import query_knowledge_graph_service

async def execute(
    query: str | None = None,
):
    """
    Query the knowledge graph for a specified repository.
    Args:
        query:
            The query string to execute against the knowledge graph in Cypher.
            The KG has the following schema:
            - (:Contributor)-[:AUTHORED]->(:Commit)
            - (:Contributor)-[:CREATED]->(:Issue)
            - (:Contributor)-[:COMMENTED]->(:IssueComment)
            - (:Contributor)-[:COMMENTED]->(:CommitComment)
            - (:Contributor)-[:PERFORMED]->(:IssueEvent)
            - (:Commit)-[:MODIFIED]->(:File)
            - (:Commit)-[:BELONGS_TO]->(:Repository)
            - (:Issue)-[:BELONGS_TO]->(:Repository)
            - (:File)-[:BELONGS_TO]->(:Repository)
            - (:IssueComment)-[:ON_ISSUE]->(:Issue)
            - (:IssueEvent)-[:ON_ISSUE]->(:Issue)
            - (:CommitComment)-[:ON_COMMIT]->(:Commit)
    """
    return await query_knowledge_graph_service(query)