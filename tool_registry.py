from tools.execute_math_calculation import execute as execute_math_calculation_execute
from tools.list_and_explore_schema import execute as list_and_explore_schema_execute
from tools.retrieve_documents import execute as retrieve_documents_execute
from tools.count_documents import execute as count_documents_execute
from tools.aggregate_collection import execute as aggregate_collection_execute
from tools.execute_ml_pipeline import execute as execute_ml_pipeline_execute
from tools.query_knowledge_graph import execute as query_knowledge_graph_execute
TOOL_REGISTRY = {
    "execute_ml_pipeline": execute_ml_pipeline_execute,
    "retrieve_documents": retrieve_documents_execute,
    "count_documents": count_documents_execute,
    "aggregate_collection": aggregate_collection_execute,
    "list_and_explore_schema": list_and_explore_schema_execute,
    "execute_math_calculation": execute_math_calculation_execute,
    "query_knowledge_graph": query_knowledge_graph_execute,
}
