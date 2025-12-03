"""
Interactive chat interface for sustainability evaluation.
"""
from pathlib import Path
from generator import SustainabilityGenerator
from evaluator import FaithfulnessEvaluator


def print_header():
    """Print application header."""
    print("\n" + "="*80)
    print("SUSTAINABILITY FRAMEWORK EVALUATION ASSISTANT")
    print("="*80)
    print("\nEvaluate business concepts against:")
    print("  • Sustainable Development Goals (SDGs)")
    print("  • UN 2030 Agenda")
    print("  • EU Taxonomy")
    print("  • Corporate Sustainability Reporting Directive (CSRD)")
    print("\nCommands:")
    print("  • Type your business concept to get evaluation")
    print("  • Type 'quit' or 'exit' to end session")
    print("  • Type 'score' to enable/disable faithfulness scoring")
    print("="*80 + "\n")


def main():
    """Run interactive chat interface."""
    print("Loading model and vector stores...")
    generator = SustainabilityGenerator()
    evaluator = FaithfulnessEvaluator(generator)
    
    print_header()
    
    show_scores = False
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!\n")
                break
            
            if query.lower() == 'score':
                show_scores = not show_scores
                status = "enabled" if show_scores else "disabled"
                print(f"\nFaithfulness scoring {status}\n")
                continue
            
            # Generate response
            if show_scores:
                print("\nGenerating and evaluating...")
                result = evaluator.evaluate_query(query)
                
                print("\nAssistant:")
                print("-" * 80)
                print(result['response'])
                print("-" * 80)
                print(f"\nSources: {', '.join(result['sources'])}")
                print(f"Faithfulness Score: {result['faithfulness_score']:.4f}")
            else:
                print("\nGenerating...")
                result = generator.generate(query)
                
                print("\nAssistant:")
                print("-" * 80)
                print(result['answer'])
                print("-" * 80)
                print(f"\nSources: {', '.join(result['sources'])}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
