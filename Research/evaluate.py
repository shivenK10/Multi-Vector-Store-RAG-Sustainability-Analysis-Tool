"""
Single-query evaluation script with faithfulness scoring.
"""
from generator import SustainabilityGenerator
from evaluator import FaithfulnessEvaluator


# Query to evaluate - modify this to test different business ideas
QUERY = "Our company wants to offer rooftop solar leasing for small and medium-sized enterprises in Eastern Europe. Evaluate how this idea aligns with the SDGs, EU Taxonomy, CSRD, and the UN 2030 Agenda."
# QUERY = "We are developing an AI-powered platform to optimize water usage in agricultural farms, reducing waste by 40% while maintaining crop yields. How does this align with sustainability frameworks?"


def main():
    """Evaluate single query and display results."""
    print("\nLoading model and vector stores...")
    generator = SustainabilityGenerator()
    evaluator = FaithfulnessEvaluator(generator)
    
    print(f"\nQuery: {QUERY}\n")
    
    # Generate and evaluate
    result = evaluator.evaluate_query(QUERY)
    
    print("Response:")
    print(result['response'])
    
    print(f"\n\nSources: {', '.join(result['sources'])}")
    print(f"Faithfulness Score: {result['faithfulness_score']:.4f}")


if __name__ == "__main__":
    main()
