import math
from services.buffer import buffer_registry

def calculate_math_subcomponent(
    operation: str, 
    data_points: list = None, 
    buffer_id: str = "latest",
    field: str = None,
    value_a: float = None, 
    value_b: float = None, 
    repo_name: str = None
):
    try:
        if not data_points and buffer_id:
            try:
                buffered_data = buffer_registry.get(buffer_id)
                if isinstance(buffered_data, list):
                    data_points = []
                    numeric_field_names = set()
                    for item in buffered_data:
                        if isinstance(item, dict):
                            numeric_field_names.update(
                                k for k, v in item.items() if isinstance(v, (int, float))
                            )

                    if not field and len(numeric_field_names) > 1:
                        return {
                            "error": (
                                f"Buffer '{buffer_id}' contains multiple numeric fields "
                                f"{sorted(numeric_field_names)}; specify which one to use "
                                "via the 'field' argument."
                            )
                        }

                    resolved_field = field if field else (
                        next(iter(numeric_field_names)) if numeric_field_names else None
                    )

                    for item in buffered_data:
                        if isinstance(item, (int, float)):
                            data_points.append(item)
                        elif isinstance(item, dict):
                            if resolved_field and resolved_field in item:
                                val = item[resolved_field]
                                if isinstance(val, (int, float)):
                                    data_points.append(val)
                            elif field:
                                return {
                                    "error": f"Field '{field}' not found in buffered item: {list(item.keys())}"
                                }
            except Exception as e:
                return {"error": f"Failed to retrieve data from buffer '{buffer_id}': {str(e)}"}

        data = [x for x in data_points if x is not None] if data_points else []

        if operation == "average":
            if not data: return {"result": 0}
            return {"result": sum(data) / len(data)}

        elif operation == "median":
            if not data: return {"result": 0}
            sorted_data = sorted(data)
            n = len(sorted_data)
            mid = n // 2
            res = (sorted_data[mid] + sorted_data[~mid]) / 2
            return {"result": res}

        elif operation == "std_dev":
            if not data or len(data) < 2: return {"result": 0}
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return {"result": math.sqrt(variance)}

        elif operation == "percentage":
            if not value_b or value_b == 0: return {"result": 0}
            return {"result": (value_a / value_b) * 100}

        elif operation == "ratio":
            if not value_b or value_b == 0: return {"result": value_a}
            return {"result": value_a / value_b}

        elif operation == "percentage_growth":
            if not value_b or value_b == 0: return {"result": 0}
            return {"result": ((value_a - value_b) / value_b) * 100}

        else:
            return {"error": f"Unknown math operation: {operation}"}

    except Exception as e:
        return {"error": f"Math engine failure: {str(e)}"}