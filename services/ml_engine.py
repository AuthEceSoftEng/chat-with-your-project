import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor
from services.buffer import buffer_registry
from ollama import embeddings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor
from services.buffer import buffer_registry
from ollama import embeddings

def execute_ml_pipeline(
    task_type: str, 
    target_column: str = None, 
    features: list = None, 
    inference_input: any = None,
    data: list = None,
    buffer_id: str = "latest",
):
    if data is None:
        try:
            data = buffer_registry.get(buffer_id)
        except Exception as e:
            raise ValueError(f"Failed to retrieve dataset from buffer '{buffer_id}': {str(e)}")

    if not data or not isinstance(data, list):
        raise ValueError("Provided or buffered dataset is empty or invalid.")

    df = pd.DataFrame(data)
    
    if not features:
        features = [col for col in df.columns if col != target_column and col != "_id"]

    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError(f"None of the provided features {features} are present in the dataset columns: {list(df.columns)}")

    if task_type == "classification":
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing or required for classification.")
        X = df[valid_features]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=100, max_depth=4)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = {"f1_score": float(f1_score(y_test, preds, average='weighted'))}
        
    elif task_type == "regression":
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing or required for regression.")
        X = df[valid_features]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = {"mae": float(mean_absolute_error(y_test, preds))}

    elif task_type == "clustering":
        combined_text = df[valid_features].fillna('').astype(str).agg(' '.join, axis=1)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_vec = vectorizer.fit_transform(combined_text)
        
        n_clusters = min(5, len(df))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = model.fit_predict(X_vec)
        
        if len(set(cluster_labels)) > 1 and len(df) > n_clusters:
            score = float(silhouette_score(X_vec, cluster_labels))
        else:
            score = 0.0
            
        metric = {"silhouette_score": score}
        
        if not inference_input:
            final_prediction = []
        else:
            df_inf = pd.DataFrame([inference_input] if isinstance(inference_input, dict) else inference_input)
            inf_text = df_inf[valid_features].fillna('').astype(str).agg(' '.join, axis=1)
            inf_vec = vectorizer.transform(inf_text)
            final_prediction = model.predict(inf_vec).tolist()
        
        return {
            "metrics": metric,
            "prediction": final_prediction
        }

    elif task_type == "semantic_similarity":
        combined_text = [str(val)[:1000] for val in df[valid_features].fillna('').astype(str).agg(' '.join, axis=1)]
        embed_model = "all_minilm:latest"
        doc_embeddings = []
        use_fallback = False

        try:
            from ollama import embeddings
            for text in combined_text:
                response = embeddings(model=embed_model, prompt=text)
                doc_embeddings.append(response['embedding'])
            doc_embeddings = np.array(doc_embeddings)
        except Exception:
            use_fallback = True

        if use_fallback or len(doc_embeddings) == 0:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            doc_matrix = vectorizer.fit_transform(combined_text).toarray()
            
            metric = {"status": "success_tfidf_fallback", "total_documents": len(df)}
            
            if not inference_input:
                sim_matrix = cosine_similarity(doc_matrix, doc_matrix)
                final_prediction = sim_matrix.tolist()
            else:
                df_inf = pd.DataFrame([inference_input] if isinstance(inference_input, dict) else inference_input)
                inf_text = [str(val)[:1000] for val in df_inf[valid_features].fillna('').astype(str).agg(' '.join, axis=1)]
                inf_matrix = vectorizer.transform(inf_text).toarray()
                
                sim_matrix = cosine_similarity(inf_matrix, doc_matrix)
                final_prediction = []
                for scores in sim_matrix:
                    top_indices = np.argsort(scores)[::-1]
                    matches = [
                        {"index": int(idx), "similarity_score": float(scores[idx]), "record": df.iloc[idx].to_dict()}
                        for idx in top_indices[:5]
                    ]
                    final_prediction.append(matches)
        else:
            metric = {"status": "success_ollama", "embedding_model": embed_model, "total_documents": len(df)}
            
            if not inference_input:
                sim_matrix = cosine_similarity(doc_embeddings, doc_embeddings)
                final_prediction = sim_matrix.tolist()
            else:
                df_inf = pd.DataFrame([inference_input] if isinstance(inference_input, dict) else inference_input)
                inf_text = [str(val)[:1000] for val in df_inf[valid_features].fillna('').astype(str).agg(' '.join, axis=1)]
                
                inf_embeddings = []
                try:
                    for text in inf_text:
                        response = embeddings(model=embed_model, prompt=text)
                        inf_embeddings.append(response['embedding'])
                    inf_embeddings = np.array(inf_embeddings)
                    sim_matrix = cosine_similarity(inf_embeddings, doc_embeddings)
                except Exception:
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                    doc_matrix = vectorizer.fit_transform(combined_text).toarray()
                    inf_matrix = vectorizer.transform(inf_text).toarray()
                    sim_matrix = cosine_similarity(inf_matrix, doc_matrix)

                final_prediction = []
                for scores in sim_matrix:
                    top_indices = np.argsort(scores)[::-1]
                    matches = [
                        {"index": int(idx), "similarity_score": float(scores[idx]), "record": df.iloc[idx].to_dict()}
                        for idx in top_indices[:5]
                    ]
                    final_prediction.append(matches)

        return {
            "metrics": metric,
            "prediction": final_prediction
        }
        
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    if not inference_input:
        final_prediction = []
    else:
        df_inf = pd.DataFrame([inference_input] if isinstance(inference_input, dict) else inference_input)
        final_prediction = model.predict(df_inf[valid_features]).tolist()
    
    return {
        "metrics": metric,
        "prediction": final_prediction
    }