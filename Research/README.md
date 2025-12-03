# Sustainability Evaluation System

RAG-based system for evaluating business concepts against sustainability frameworks using Llama 3.2 3B.

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Retriever (retriever.py)          │
│   - FAISS vector stores              │
│   - all-MiniLM-L6-v2 embeddings     │
│   - Multi-framework search           │
└────────┬────────────────────────────┘
         │ Retrieved contexts
         ▼
┌─────────────────────────────────────┐
│   Generator (generator.py)           │
│   - Llama 3.2 3B Instruct           │
│   - Structured prompts               │
│   - Context formatting               │
└────────┬────────────────────────────┘
         │ Generated response
         ▼
┌─────────────────────────────────────┐
│   Evaluator (evaluator.py)          │
│   - Llama 3.2 3B as judge           │
│   - Faithfulness scoring             │
└────────┬────────────────────────────┘
         │
         ▼
    Final Output
```

## Components

### 1. **retriever.py** - Context Retrieval
- Manages 4 FAISS vector stores (SDGs, UN 2030, EU Taxonomy, CSRD)
- Encodes queries with sentence-transformers
- Retrieves top-k most relevant chunks with metadata
- Supports framework-specific or cross-framework search

### 2. **generator.py** - Response Generation
- Loads Llama 3.2 3B Instruct model
- Creates structured prompts for sustainability evaluation
- Formats retrieved context by framework
- Generates responses with explicit citations and framework tags
- Trims excessive output (max 5 key gaps)

### 3. **evaluator.py** - Faithfulness Evaluation
- Uses same Llama 3.2 3B model as judge
- Scores faithfulness (0-1) of generated responses
- Truncates contexts to fit token budget
- Returns parsed numeric score

### 4. **evaluate.py** - Single Query Evaluation
- Simple script for testing individual queries
- Edit `QUERY` variable to test different concepts
- Displays response with sources and faithfulness score

### 5. **chat.py** - Interactive Interface
- Conversational interface for multiple queries
- Toggle faithfulness scoring with `score` command
- Displays formatted responses with sources

## Usage

### Single Query Evaluation
```bash
# Edit QUERY in evaluate.py, then run:
python3 evaluate.py
```

### Interactive Chat
```bash
python3 chat.py
```

### Programmatic Use
```python
from generator import SustainabilityGenerator
from evaluator import FaithfulnessEvaluator

# Initialize
generator = SustainabilityGenerator()
evaluator = FaithfulnessEvaluator(generator)

# Evaluate query
result = evaluator.evaluate_query(
    "Your business concept here"
)

print(result['response'])
print(f"Faithfulness: {result['faithfulness_score']:.4f}")
```

## Output Format

Responses follow strict structure:

1. **SDG Mapping**: Relevant SDGs with quoted context and [SDGs] tag
2. **Policy & Standards Alignment**: Policies with quotes and [Framework] tags
3. **EU Taxonomy Eligibility**: Quoted clauses with [EU Taxonomy] tag
4. **CSRD/ESRS Relevance**: Quoted content with [CSRD] tag
5. **Key Gaps**: Max 5 bullets listing missing information

## Model Configuration

- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Generation**: Greedy decoding (do_sample=False) for faithfulness
- **Max Tokens**: 1500
- **Repetition Penalty**: 1.05
- **Context Retrieval**: 25 chunks (6 per framework max)
- **Min Similarity**: 0.15

## Dependencies

- transformers
- sentence-transformers
- faiss-cpu (or faiss-gpu)
- langchain-huggingface
- torch

## Framework Data

Requires pre-built FAISS indices in `vector_stores/`:
- `sdgs_faiss.index` + `sdgs_metadata.pkl`
- `un_2030_agenda_faiss.index` + `un_2030_agenda_metadata.pkl`
- `eu_taxonomy_user_guide_faiss.index` + `eu_taxonomy_user_guide_metadata.pkl`
- `csrd_fact_sheet_faiss.index` + `csrd_fact_sheet_metadata.pkl`

Build these using `build_faiss.py` with your JSON source files.

## Performance

- **Model Load**: ~5 seconds (MPS/Metal)
- **Query Retrieval**: <1 second
- **Response Generation**: 5-10 seconds
- **Faithfulness Eval**: 2-3 seconds
- **Total**: ~10-15 seconds per query

## Notes

- Use `model_handler.py` for model loading with quantization support
- Faithfulness scores: 0.75+ = good, 0.5-0.75 = moderate, <0.5 = review needed
- Key gaps limited to 5 items to prevent model rambling
- Framework tags required in all citations for traceability
