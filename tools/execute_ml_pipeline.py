from services.ml_engine import execute_ml_pipeline

async def execute(
    task_type: str,
    data: list[dict] | None = None,
    target_column: str | None = None,
    features: list[str] | None = None,
    inference_input: dict | list[dict] | None = None,
    buffer_id: str | None = None,
):
    """
    Execute a machine learning pipeline for classification, clustering, or
    regression. Use buffer_id to consume a dataset returned by a database or
    knowledge graph tool. A buffer_id such as buf_abc123 is an in-memory
    handle, never a MongoDB collection name.
    Args:
        data:
            Optional list of dictionaries representing the dataset. Omit it
            when using buffer_id.
        task_type:
            The type of ML task to perform. Options are "classification", "clustering", "regression" or "semantic_similarity".
        target_column:
            The name of the column in the dataset that contains the target variable.
        features:
            A list of column names to be used as features for training the model.
        inference_input:
            A dictionary or list of dictionaries containing input data for making predictions.
        buffer_id:
            Optional in-memory dataset handle returned by a data tool. Do not
            pass it as a collection name.
    """
    return execute_ml_pipeline(
        data=data,
        task_type=task_type,
        target_column=target_column,
        features=features,
        inference_input=inference_input,
        buffer_id=buffer_id or "latest",
)