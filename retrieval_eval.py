"""
Retrieval Quality Evaluation Module

This module provides functions to evaluate the retrieval quality of 
MultiVectorStoreRAGPipeline using standard Information Retrieval metrics:
- Precision@k
- Recall@k

The evaluation uses a JSONL file with query-document relevance annotations.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from statistics import mean

from generation_pipeline import MultiVectorStoreRAGPipeline
from logger import Logger


def get_eval_logger() -> Logger:
    """
    Create a Logger instance for evaluation tasks.
    """
    logs_dir = Path("Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = logs_dir / "retrieval_eval.log"

    logger = Logger(
        name="RetrievalEvaluator",
        log_file_needed=True,
        log_file=str(log_file_path),
        level="DEV",
    )
    return logger


def load_retrieval_eval(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from a JSONL file.
    
    Args:
        path: Path to the JSONL evaluation file
        
    Returns:
        List of evaluation examples, each with:
        - "query": natural language question
        - "relevant_doc_ids": list of relevant document IDs
        
    Raises:
        FileNotFoundError: If the evaluation file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    logger = get_eval_logger()
    eval_path = Path(path)
    
    if not eval_path.exists():
        logger.error(f"Evaluation file not found: {path}")
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    
    logger.info(f"Loading evaluation data from: {path}")
    
    eval_data = []
    try:
        with eval_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    example = json.loads(line)
                    
                    # Validate required fields
                    if "query" not in example:
                        logger.warning(f"Line {line_num}: Missing 'query' field, skipping")
                        continue
                    if "relevant_doc_ids" not in example:
                        logger.warning(f"Line {line_num}: Missing 'relevant_doc_ids' field, skipping")
                        continue
                    if not isinstance(example["relevant_doc_ids"], list):
                        logger.warning(f"Line {line_num}: 'relevant_doc_ids' must be a list, skipping")
                        continue
                        
                    eval_data.append(example)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON, skipping - {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Failed to read evaluation file: {str(e)}")
        raise
    
    logger.info(f"Loaded {len(eval_data)} valid evaluation examples")
    return eval_data


def precision_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Compute Precision@k for retrieval evaluation.
    
    Precision@k = (# of relevant doc IDs in the first k predicted IDs) / k
    
    Args:
        pred_ids: List of predicted document IDs (in ranked order)
        gold_ids: List of ground truth relevant document IDs
        k: Number of top predictions to consider
        
    Returns:
        Precision@k score (float between 0.0 and 1.0)
        Returns 0.0 if k == 0
    """
    if k == 0:
        return 0.0
    
    top_k_preds = pred_ids[:k]
    
    pred_set = set(top_k_preds)
    gold_set = set(gold_ids)
    
    # Count relevant items in top-k
    relevant_in_topk = len(pred_set.intersection(gold_set))
    
    return relevant_in_topk / k


def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Compute Recall@k for retrieval evaluation.
    
    Recall@k = (# of relevant doc IDs in the first k predicted IDs) / (total # relevant items)
    
    Args:
        pred_ids: List of predicted document IDs (in ranked order)
        gold_ids: List of ground truth relevant document IDs
        k: Number of top predictions to consider
        
    Returns:
        Recall@k score (float between 0.0 and 1.0)
        Returns 0.0 if there are no gold_ids
    """
    if len(gold_ids) == 0:
        return 0.0
    
    top_k_preds = pred_ids[:k]
    
    pred_set = set(top_k_preds)
    gold_set = set(gold_ids)
    
    # Count relevant items in top-k
    relevant_in_topk = len(pred_set.intersection(gold_set))
    
    return relevant_in_topk / len(gold_ids)


def extract_doc_ids_from_retrieval(all_docs: Dict[str, List], logger: Logger) -> List[str]:
    """
    Extract document IDs from MultiVectorStoreRAGPipeline retrieval results.
    
    Args:
        all_docs: Dictionary mapping source names to lists of Documents
        logger: Logger instance for tracking
        
    Returns:
        List of document IDs in the order they were retrieved
    """
    doc_ids = []
    
    for source_name, docs in all_docs.items():
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                doc_ids.append(doc_id)
            else:
                chunk_id = doc.metadata.get("chunk_id")
                filename = doc.metadata.get("filename", source_name)
                if chunk_id is not None:
                    fallback_id = f"{filename}_chunk_{chunk_id}"
                else:
                    # Use a hash of the content as last resort
                    import hashlib
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                    fallback_id = f"{filename}_{content_hash}"
                
                doc_ids.append(fallback_id)
                logger.debug(f"Using fallback doc_id: {fallback_id}")
    
    logger.debug(f"Extracted {len(doc_ids)} document IDs from retrieval results")
    return doc_ids


def evaluate_single_query(
    pipeline: MultiVectorStoreRAGPipeline,
    query: str,
    relevant_doc_ids: List[str],
    k: int,
    logger: Logger
) -> Dict[str, float]:
    """
    Evaluate a single query against the retrieval pipeline.
    
    Args:
        pipeline: The RAG pipeline to evaluate
        query: The query string
        relevant_doc_ids: List of relevant document IDs for this query
        k: Number of top documents to retrieve
        logger: Logger instance
        
    Returns:
        Dictionary with precision@k and recall@k scores
    """
    try:
        k_per_source = max(1, k // len(pipeline.vectorstores))
        all_docs = pipeline.retrieve_documents(query, k_per_source=k_per_source)
        
        retrieved_doc_ids = extract_doc_ids_from_retrieval(all_docs, logger)
        
        precision = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        recall = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        
        logger.debug(f"Query: '{query[:50]}...' | P@{k}: {precision:.3f} | R@{k}: {recall:.3f}")
        
        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "retrieved_count": len(retrieved_doc_ids),
            "relevant_count": len(relevant_doc_ids)
        }
        
    except Exception as e:
        logger.error(f"Failed to evaluate query '{query[:50]}...': {str(e)}")
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "retrieved_count": 0,
            "relevant_count": len(relevant_doc_ids)
        }


def evaluate_retriever(
    pipeline: MultiVectorStoreRAGPipeline,
    eval_path: str,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate the retriever using Precision@k and Recall@k
    on queries stored in `eval_path`.
    
    Args:
        pipeline: The MultiVectorStoreRAGPipeline to evaluate
        eval_path: Path to the JSONL evaluation file
        k: Number of top documents to consider (default: 5)
        
    Returns:
        Dictionary with macro-averaged metrics and detailed results:
        - "precision_at_k": Macro-averaged Precision@k
        - "recall_at_k": Macro-averaged Recall@k
        - "num_queries": Number of queries evaluated
        - "k": The k value used
        - "per_query_results": List of per-query results
        - "summary_stats": Additional statistics
    """
    logger = get_eval_logger()
    logger.info(f"Starting retrieval evaluation with k={k}")
    logger.info(f"Evaluation file: {eval_path}")
    
    try:
        eval_data = load_retrieval_eval(eval_path)
    except Exception as e:
        logger.error(f"Failed to load evaluation data: {str(e)}")
        return {
            "error": str(e),
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "num_queries": 0,
            "k": k
        }
    
    if not eval_data:
        logger.warning("No valid evaluation examples found")
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "num_queries": 0,
            "k": k,
            "per_query_results": []
        }
    
    # Evaluate each query
    per_query_results = []
    precision_scores = []
    recall_scores = []
    
    logger.info(f"Evaluating {len(eval_data)} queries...")
    
    for i, example in enumerate(eval_data, 1):
        query = example["query"]
        relevant_doc_ids = example["relevant_doc_ids"]
        
        logger.debug(f"Evaluating query {i}/{len(eval_data)}: '{query[:50]}...'")
        
        # Evaluate this query
        result = evaluate_single_query(pipeline, query, relevant_doc_ids, k, logger)
        
        per_query_results.append({
            "query": query,
            "relevant_doc_ids": relevant_doc_ids,
            **result
        })
        
        precision_scores.append(result["precision_at_k"])
        recall_scores.append(result["recall_at_k"])
    
    avg_precision = mean(precision_scores) if precision_scores else 0.0
    avg_recall = mean(recall_scores) if recall_scores else 0.0
    
    total_relevant_docs = sum(len(ex["relevant_doc_ids"]) for ex in eval_data)
    avg_relevant_per_query = total_relevant_docs / len(eval_data) if eval_data else 0
    
    results = {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "num_queries": len(eval_data),
        "k": k,
        "per_query_results": per_query_results,
        "summary_stats": {
            "total_relevant_docs": total_relevant_docs,
            "avg_relevant_per_query": avg_relevant_per_query,
            "min_precision": min(precision_scores) if precision_scores else 0.0,
            "max_precision": max(precision_scores) if precision_scores else 0.0,
            "min_recall": min(recall_scores) if recall_scores else 0.0,
            "max_recall": max(recall_scores) if recall_scores else 0.0,
        }
    }
    
    logger.info("=" * 60)
    logger.info("RETRIEVAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Number of queries evaluated: {results['num_queries']}")
    logger.info(f"k (top documents considered): {results['k']}")
    logger.info(f"Macro-averaged Precision@{k}: {avg_precision:.4f}")
    logger.info(f"Macro-averaged Recall@{k}: {avg_recall:.4f}")
    logger.info(f"Total relevant documents: {total_relevant_docs}")
    logger.info(f"Average relevant docs per query: {avg_relevant_per_query:.2f}")
    logger.info("=" * 60)
    
    return results


def create_sample_eval_file(output_path: str = "eval_data/retrieval_eval.jsonl") -> None:
    """
    Create a sample evaluation file for demonstration purposes.
    
    Args:
        output_path: Path where to create the sample file
    """
    logger = get_eval_logger()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data = [
        {
            "query": "sustainable energy solutions for manufacturing",
            "relevant_doc_ids": ["sdg_7_energy", "sdg_9_innovation", "eu_taxonomy_energy"]
        },
        {
            "query": "carbon emission reduction strategies",
            "relevant_doc_ids": ["sdg_13_climate", "csrd_emissions", "un_2030_carbon"]
        },
        {
            "query": "circular economy and waste management",
            "relevant_doc_ids": ["sdg_12_consumption", "eu_taxonomy_circular", "csrd_waste"]
        },
        {
            "query": "renewable energy infrastructure",
            "relevant_doc_ids": ["sdg_7_renewable", "eu_taxonomy_renewable"]
        },
        {
            "query": "biodiversity conservation in business",
            "relevant_doc_ids": ["sdg_15_biodiversity", "eu_taxonomy_biodiversity"]
        }
    ]
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for example in sample_data:
                f.write(json.dumps(example) + "\n")
        
        logger.info(f"Created sample evaluation file: {output_path}")
        logger.info(f"Contains {len(sample_data)} example queries")
        
    except Exception as e:
        logger.error(f"Failed to create sample evaluation file: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Demo script showing how to use the retrieval evaluation module.
    """
    print("=" * 60)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 60)
    
    print("\n1. Creating sample evaluation data...")
    sample_path = "eval_data/retrieval_eval.jsonl"
    create_sample_eval_file(sample_path)
    
    print("\n2. Loading evaluation data...")
    eval_data = load_retrieval_eval(sample_path)
    print(f"   Loaded {len(eval_data)} evaluation examples")
    print(f"   Sample query: '{eval_data[0]['query']}'")
    print(f"   Sample relevant docs: {eval_data[0]['relevant_doc_ids']}")
    
    print("\n3. Demonstrating metric functions...")
    pred_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    gold_ids = ["doc1", "doc3", "doc6"]
    k = 3
    
    precision = precision_at_k(pred_ids, gold_ids, k)
    recall = recall_at_k(pred_ids, gold_ids, k)
    
    print(f"   Example: pred_ids = {pred_ids[:k]}")
    print(f"   Example: gold_ids = {gold_ids}")
    print(f"   Precision@{k} = {precision:.3f}")
    print(f"   Recall@{k} = {recall:.3f}")
    
    print(f"\n4. To run full evaluation, use:")
    print(f"   from retrieval_eval import evaluate_retriever")
    print(f"   from generation_pipeline import MultiVectorStoreRAGPipeline")
    print(f"   pipeline = MultiVectorStoreRAGPipeline()")
    print(f"   results = evaluate_retriever(pipeline, '{sample_path}', k=5)")
    print(f"   print(f'Precision@5: {{results[\"precision_at_k\"]:.4f}}')")
    print(f"   print(f'Recall@5: {{results[\"recall_at_k\"]:.4f}}')")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE - Evaluation data created at:", sample_path)
    print("=" * 60)