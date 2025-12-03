"""
Evaluation module for measuring faithfulness using Llama 3.2 3B as judge.
"""
import re
from typing import List, Dict
from generator import SustainabilityGenerator


class FaithfulnessEvaluator:
    """Evaluates faithfulness of generated responses using LLM-as-judge."""
    
    def __init__(self, generator: SustainabilityGenerator):
        """
        Initialize evaluator with generator instance.
        
        Args:
            generator: SustainabilityGenerator with loaded model
        """
        self.generator = generator
        self.pipe = generator.pipe
    
    def _truncate_contexts(
        self,
        contexts: List[str],
        max_chars: int = 5000,
        max_chunks: int = 10
    ) -> List[str]:
        """Truncate contexts to fit within token budget."""
        out = []
        total = 0
        
        for text in contexts[:max_chunks]:
            text = text.strip()
            if not text:
                continue
            
            remaining = max_chars - total
            if remaining <= 0:
                break
            
            if len(text) > remaining:
                if remaining > 50:
                    out.append(text[:remaining])
                break
            
            out.append(text)
            total += len(text) + 1
        
        return out
    
    def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate faithfulness using Llama 3.2 3B as judge.
        
        Args:
            query: Original user query
            response: Generated response to evaluate
            contexts: Retrieved context chunks
        
        Returns:
            Faithfulness score between 0 and 1
        """
        try:
            if not contexts:
                print("Warning: No contexts provided; returning 0.0")
                return 0.0
            
            # Truncate contexts
            truncated = self._truncate_contexts(contexts, max_chars=5000, max_chunks=10)
            joined_context = "\n\n".join(
                f"[C{i+1}]\n{c}" for i, c in enumerate(truncated)
            )
            
            # Create judge prompt
            eval_prompt = (
                "You are evaluating faithfulness. Return ONLY a number between 0.0 and 1.0.\n"
                "1.0 = all claims supported by context\n"
                "0.5 = some claims supported\n"
                "0.0 = not supported\n\n"
                f"Question: {query}\n\n"
                f"Answer: {response[:800]}...\n\n"
                f"Context: {joined_context[:2000]}...\n\n"
                "Score (0.0 to 1.0): "
            )
            
            # Generate score
            outputs = self.pipe(eval_prompt, max_new_tokens=16, do_sample=False)
            
            if not outputs:
                print("Warning: Judge returned no output; returning 0.0")
                return 0.0
            
            generated = outputs[0].get("generated_text", "").strip()
            
            # Parse numeric score - try multiple patterns
            match = re.search(r"([01](?:\.\d{1,4})?)", generated)
            if not match:
                # Try finding any decimal
                match = re.search(r"(\d\.\d+)", generated)
            
            if not match:
                print(f"Warning: Could not parse score from: {generated!r}")
                # Default to 0.5 instead of 0.0 if parsing fails
                return 0.5
            
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return 0.0
    
    def evaluate_query(self, query: str, top_k: int = 25) -> Dict:
        """
        Full pipeline: generate response and evaluate faithfulness.
        
        Args:
            query: Business concept to evaluate
            top_k: Number of contexts to retrieve
        
        Returns:
            Dict with 'query', 'response', 'sources', 'faithfulness_score'
        """
        # Generate response
        result = self.generator.generate(query, top_k=top_k)
        
        # Extract contexts
        contexts = [chunk['text'] for chunk in result['retrieved_chunks']]
        
        # Evaluate faithfulness
        faithfulness = self.evaluate(query, result['answer'], contexts)
        
        return {
            "query": query,
            "response": result['answer'],
            "sources": result['sources'],
            "faithfulness_score": faithfulness,
            "num_contexts": len(contexts)
        }


if __name__ == "__main__":
    from generator import SustainabilityGenerator
    
    # Initialize
    print("Loading model and evaluator...")
    generator = SustainabilityGenerator()
    evaluator = FaithfulnessEvaluator(generator)
    
    # Test query
    test_query = "Our company offers electric vehicle charging stations powered by solar energy for commercial buildings."
    
    print(f"\nEvaluating query: {test_query}\n")
    
    result = evaluator.evaluate_query(test_query)
    
    print("Response:")
    print(result['response'])
    print(f"\n\nSources: {', '.join(result['sources'])}")
    print(f"Contexts retrieved: {result['num_contexts']}")
    print(f"Faithfulness Score: {result['faithfulness_score']:.4f}")
