from typing import Literal
from services.math_engine import calculate_math_subcomponent


async def execute(
    repo_name: str,
    operation: Literal[
        "average",
        "median",
        "std_dev",
        "percentage",
        "ratio",
        "percentage_growth",
    ],
    data_points: list[float] | None = None,
    buffer_id: str | None = None,
    field: str | None = None,
    value_a: float | None = None,
    value_b: float | None = None,
):
    """
    A mathematical engine that performs operations on list arrays or numbers.
    Use buffer_id to consume the dataset returned by retrieve_documents,
    aggregate_collection, or query_knowledge_graph. A buffer_id such as
    buf_abc123 is an in-memory handle, never a MongoDB collection name.
    Args:
        operation:
            The type of mathematical or statistical operation to execute.
        data_points:
            An array of numbers to calculate statistics over.
        buffer_id:
            Optional in-memory dataset handle returned by a data tool. Use
            this instead of calling a data tool again or using the handle as a
            collection name. The latest buffer is used when omitted.
        field:
            Required when the buffered dataset contains documents/dicts with
            more than one numeric field (e.g. a time index and a count). Set
            this to the exact field name (as seen in the data tool's result)
            that should be used for the calculation. If omitted and the
            buffer is unambiguous (only one numeric field), it is inferred
            automatically; otherwise the call returns an error listing the
            available numeric field names instead of guessing.
        value_a:
            First isolated value for direct 2-number metrics.
        value_b:
            Second isolated value for direct 2-number metrics.
    """
    return calculate_math_subcomponent(
        repo_name=repo_name,
        operation=operation,
        data_points=data_points,
        buffer_id=buffer_id or "latest",
        field=field,
        value_a=value_a,
        value_b=value_b,
    )