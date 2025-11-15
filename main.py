from generation_pipeline import MultiVectorStoreRAGPipeline

def main():
    # Configuration - modify these values as needed
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    quantize = False
    single_idea = None  # Set to a string if you want to analyze just one idea and exit
    # index_dirs will use all available vector stores by default

    pipeline = MultiVectorStoreRAGPipeline(
        llm_model_name=model_name,
        quantize=quantize,
    )

    if single_idea:
        print("\n=== Company Idea ===")
        print(single_idea)
        print("\n=== SDG Analysis (RAG) ===")
        output = pipeline.generate(single_idea)
        print(output)
        return

    print("SDG RAG Chatbot")
    print("Type your company idea and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            idea = input("Your idea> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if idea.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not idea:
            continue

        print("\nThinking...\n")
        output = pipeline.generate(idea)
        print(output)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
