from transformers import TapasTokenizer, TapasModel
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
from typing import Tuple, Optional

def get_table_embedding_batch(table, model, tokenizer, device='cpu', method='multi_query', batch_size=200):
    """
    Process large tables in batches to generate embeddings using all data
    
    Args:
        table: DataFrame table
        model: TAPAS model
        tokenizer: TAPAS tokenizer
        device: Computing device
        method: Embedding method
        batch_size: Number of rows per batch
    
    Returns:
        Table embedding vector
    """
    if len(table) <= batch_size:
        return get_table_embedding(table, model, tokenizer, device, method)
    
    num_batches = (len(table) + batch_size - 1) // batch_size
    batch_embeddings = []
    
    print(f"  Processing {len(table)} rows in {num_batches} batches (batch_size={batch_size})...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(table))
        batch_table = table.iloc[start_idx:end_idx].reset_index(drop=True)
        
        batch_emb = get_table_embedding(batch_table, model, tokenizer, device, method)
        batch_embeddings.append(batch_emb)
        
        if (i + 1) % 5 == 0 or (i + 1) == num_batches:
            print(f"    Processed {i + 1}/{num_batches} batches...")
    
    table_embedding = torch.stack(batch_embeddings).mean(dim=0)
    return table_embedding

def get_table_embedding(table, model, tokenizer, device='cpu', method='multi_query'):
    """
    Get table embedding vector representation
    
    Args:
        table: DataFrame table
        model: TAPAS model
        tokenizer: TAPAS tokenizer
        device: Computing device
        method: Embedding method
            - 'cls': Use only CLS token (original method)
            - 'mean_pool': Use mean pooling of all tokens
            - 'multi_query': Use multiple queries to generate multiple embeddings and average
            - 'row_level': Row-level embedding
    """
    if method == 'cls':
        query = "describe this table"
        inputs = tokenizer(
            table=table, 
            queries=[query], 
            padding="max_length", 
            return_tensors="pt",
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            table_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return table_embedding
    
    elif method == 'mean_pool':
        query = "describe this table"
        inputs = tokenizer(
            table=table, 
            queries=[query], 
            padding="max_length", 
            return_tensors="pt",
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state[0]
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                mask = attention_mask[0].unsqueeze(-1)  # [seq_len, 1]
                masked_hidden = hidden_states * mask
                table_embedding = masked_hidden.sum(dim=0) / mask.sum()
            else:
                table_embedding = hidden_states.mean(dim=0)
        return table_embedding
    
    elif method == 'multi_query':
        queries = [
            "describe this table",
            "what are the main features of this table",
            "summarize the content of this table",
            "what information does this table contain"
        ]
        embeddings = []
        for query in queries:
            inputs = tokenizer(
                table=table, 
                queries=[query], 
                padding="max_length", 
                return_tensors="pt",
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(emb)
        table_embedding = torch.stack(embeddings).mean(dim=0)
        return table_embedding
    
    elif method == 'row_level':
        query = "describe this row"
        row_embeddings = []
        for idx, row in table.iterrows():
            row_df = pd.DataFrame([row], columns=table.columns)
            inputs = tokenizer(
                table=row_df, 
                queries=[query], 
                padding="max_length", 
                return_tensors="pt",
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                row_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
                row_embeddings.append(row_emb)
        table_embedding = torch.stack(row_embeddings).mean(dim=0)
        return table_embedding
    
    else:
        raise ValueError(f"Unknown method: {method}")

def smart_sample_tables(table1: pd.DataFrame, table2: pd.DataFrame, max_rows: int, random_seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Intelligently sample two tables, ensuring consistent row counts and matching sampling positions
    
    Args:
        table1: First table
        table2: Second table
        max_rows: Maximum number of rows
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled two tables
    """
    n1, n2 = len(table1), len(table2)
    
    if n1 <= max_rows and n2 <= max_rows:
        return table1.copy(), table2.copy()
    
    target_rows = min(min(n1, n2), max_rows)
    
    if n1 == n2:
        front_size = target_rows // 4
        middle_size = target_rows // 2
        back_size = target_rows - front_size - middle_size
        
        front = table1.iloc[:front_size], table2.iloc[:front_size]
        middle_start = n1 // 2 - middle_size // 2
        middle = (table1.iloc[middle_start:middle_start + middle_size], 
                  table2.iloc[middle_start:middle_start + middle_size])
        back = table1.iloc[-back_size:], table2.iloc[-back_size:]
        
        sampled1 = pd.concat([front[0], middle[0], back[0]], ignore_index=True)
        sampled2 = pd.concat([front[1], middle[1], back[1]], ignore_index=True)
        
        return sampled1, sampled2
    
    else:
        ratio1 = target_rows / n1 if n1 > max_rows else 1.0
        ratio2 = target_rows / n2 if n2 > max_rows else 1.0
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        def proportional_sample(df, target_size, original_size):
            if len(df) <= target_size:
                return df.copy()
            
            front_size = target_size // 4
            middle_size = target_size // 2
            back_size = target_size - front_size - middle_size
            
            front = df.iloc[:front_size]
            middle_start = original_size // 2 - middle_size // 2
            middle = df.iloc[middle_start:middle_start + middle_size]
            back = df.iloc[-back_size:]
            
            return pd.concat([front, middle, back], ignore_index=True)
        
        sampled1 = proportional_sample(table1, target_rows, n1)
        sampled2 = proportional_sample(table2, target_rows, n2)
        
        min_len = min(len(sampled1), len(sampled2))
        sampled1 = sampled1.iloc[:min_len]
        sampled2 = sampled2.iloc[:min_len]
        
        return sampled1, sampled2

def compute_similarity(embedding1, embedding2, metrics=['cosine']):
    """
    Compute similarity between two embedding vectors
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metrics: List of similarity metrics, options: 'cosine', 'euclidean', 'manhattan', 'dot'
    
    Returns:
        Scalar if only one metric, dictionary if multiple metrics
    """
    if embedding1.device != embedding2.device:
        embedding2 = embedding2.to(embedding1.device)
    
    results = {}
    
    if 'cosine' in metrics:
        cosine_sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1)
        results['cosine'] = cosine_sim.item()
    
    if 'euclidean' in metrics:
        euclidean_dist = torch.norm(embedding1 - embedding2).item()
        results['euclidean_sim'] = 1 / (1 + euclidean_dist)
    
    if 'manhattan' in metrics:
        manhattan_dist = torch.norm(embedding1 - embedding2, p=1).item()
        results['manhattan_sim'] = 1 / (1 + manhattan_dist)
    
    if 'dot' in metrics:
        dot_product = torch.dot(embedding1, embedding2).item()
        norm1 = torch.norm(embedding1).item()
        norm2 = torch.norm(embedding2).item()
        if norm1 > 0 and norm2 > 0:
            results['dot_norm'] = dot_product / (norm1 * norm2)
        else:
            results['dot_norm'] = 0.0
    
    if len(results) == 1:
        return list(results.values())[0]
    return results

def main(dirty_csv_path, clean_csv_path, embedding_method='multi_query', similarity_metrics=['cosine', 'euclidean'], 
         max_rows=200, use_batch_processing=True, batch_size=200):
    """
    Compute semantic similarity between two tables
    
    Args:
        dirty_csv_path: Path to repaired table
        clean_csv_path: Path to clean table
        embedding_method: Embedding method, options: 'cls', 'mean_pool', 'multi_query', 'row_level'
            - 'cls': Original method, use only CLS token (fastest, but less discriminative)
            - 'mean_pool': Use mean pooling (more sensitive, recommended)
            - 'multi_query': Use multiple queries (most discriminative, recommended for distinguishing repair methods)
            - 'row_level': Row-level embedding (slowest, but finest granularity)
        similarity_metrics: List of similarity metrics, options: 'cosine', 'euclidean', 'manhattan', 'dot'
        max_rows: Maximum row limit, tables exceeding this will be sampled (default 200, TAPAS model limit)
        use_batch_processing: Whether to use batch processing (True=use all data, False=sample partial data)
        batch_size: Number of rows per batch when batch processing (default 200)
    """
    use_cpu = max_rows > 300
    if use_cpu:
        device = torch.device('cpu')
        print(f"Using device: {device} (forced CPU for large tables)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    print(f"Embedding method: {embedding_method}")
    print(f"Similarity metrics: {similarity_metrics}")
    
    model_name = "google/tapas-base-finetuned-wtq"
    print("\nLoading TAPAS model and tokenizer...")
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    print(f"\nReading CSV files...")
    print(f"Reading: {dirty_csv_path}")
    table_dirty = pd.read_csv(dirty_csv_path)
    print(f"  Shape: {table_dirty.shape}")
    
    print(f"Reading: {clean_csv_path}")
    table_clean = pd.read_csv(clean_csv_path)
    print(f"  Shape: {table_clean.shape}")
    
    if 'Unnamed: 0' in table_dirty.columns:
        table_dirty = table_dirty.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in table_clean.columns:
        table_clean = table_clean.drop(columns=['Unnamed: 0'])
    
    original_dirty_rows = len(table_dirty)
    original_clean_rows = len(table_clean)
    
    if use_batch_processing:
        if original_dirty_rows > max_rows or original_clean_rows > max_rows:
            print(f"\nUsing batch processing to handle large tables...")
            print(f"  Original sizes - Dirty: {original_dirty_rows} rows, Clean: {original_clean_rows} rows")
            print(f"  Batch size: {batch_size} rows")
    else:
        if original_dirty_rows > max_rows or original_clean_rows > max_rows:
            print(f"\nWarning: Tables are too large for TAPAS model (>{max_rows} rows).")
            print(f"Sampling {max_rows} rows for processing...")
            print(f"  Original sizes - Dirty: {original_dirty_rows} rows, Clean: {original_clean_rows} rows")
            
            table_dirty, table_clean = smart_sample_tables(table_dirty, table_clean, max_rows, random_seed=42)
            print(f"  After sampling - Dirty: {table_dirty.shape}, Clean: {table_clean.shape}")
    
    def convert_to_string_table(df):
        """Convert all DataFrame columns to string and handle NaN values"""
        df_str = df.copy()
        for col in df_str.columns:
            df_str[col] = df_str[col].astype(str)
            df_str[col] = df_str[col].replace(['nan', 'NaN', 'None', 'null'], '')
        return df_str
    
    print("\nConverting tables to string format...")
    table_dirty = convert_to_string_table(table_dirty)
    table_clean = convert_to_string_table(table_clean)
    
    print(f"\nGenerating embeddings for dirty table (method: {embedding_method})...")
    if use_batch_processing and (original_dirty_rows > max_rows or original_clean_rows > max_rows):
        embedding_dirty = get_table_embedding_batch(table_dirty, model, tokenizer, device, method=embedding_method, batch_size=batch_size)
    else:
        embedding_dirty = get_table_embedding(table_dirty, model, tokenizer, device, method=embedding_method)
    
    print(f"Generating embeddings for clean table (method: {embedding_method})...")
    if use_batch_processing and (original_dirty_rows > max_rows or original_clean_rows > max_rows):
        embedding_clean = get_table_embedding_batch(table_clean, model, tokenizer, device, method=embedding_method, batch_size=batch_size)
    else:
        embedding_clean = get_table_embedding(table_clean, model, tokenizer, device, method=embedding_method)
    
    print(f"\nComputing semantic similarity (metrics: {similarity_metrics})...")
    similarity_results = compute_similarity(embedding_dirty, embedding_clean, metrics=similarity_metrics)
    
    print("\n" + "="*60)
    print("TABLE SEMANTIC SIMILARITY RESULTS")
    print("="*60)
    print(f"\nDirty Table ({os.path.basename(dirty_csv_path)}):")
    print(f"  - Original shape: {original_dirty_rows} rows × {table_dirty.shape[1]} columns")
    print(f"  - Processed shape: {table_dirty.shape[0]} rows × {table_dirty.shape[1]} columns")
    print(f"  - Columns: {', '.join(table_dirty.columns.tolist())}")
    if use_batch_processing and original_dirty_rows > max_rows:
        print(f"  - Processing mode: Batch processing (using all {original_dirty_rows} rows)")
    elif not use_batch_processing and original_dirty_rows > max_rows:
        print(f"  - Processing mode: Sampling ({table_dirty.shape[0]}/{original_dirty_rows} rows used)")
    
    print(f"\nClean Table ({os.path.basename(clean_csv_path)}):")
    print(f"  - Original shape: {original_clean_rows} rows × {table_clean.shape[1]} columns")
    print(f"  - Processed shape: {table_clean.shape[0]} rows × {table_clean.shape[1]} columns")
    print(f"  - Columns: {', '.join(table_clean.columns.tolist())}")
    if use_batch_processing and original_clean_rows > max_rows:
        print(f"  - Processing mode: Batch processing (using all {original_clean_rows} rows)")
    elif not use_batch_processing and original_clean_rows > max_rows:
        print(f"  - Processing mode: Sampling ({table_clean.shape[0]}/{original_clean_rows} rows used)")
    
    print(f"\n" + "-"*60)
    print("SIMILARITY SCORES:")
    
    if isinstance(similarity_results, dict):
        for metric, score in similarity_results.items():
            print(f"  {metric.upper()}: {score:.6f} ({score * 100:.4f}%)")
        main_similarity = similarity_results.get('cosine', list(similarity_results.values())[0])
    else:
        main_similarity = similarity_results
        print(f"  COSINE: {main_similarity:.6f} ({main_similarity * 100:.4f}%)")
    

    
    return similarity_results



