# Multi-Vector Store RAG Sustainability Analysis Tool

A comprehensive AI-powered sustainability analysis system that evaluates company ideas against multiple authoritative frameworks using adaptive document retrieval and explainable AI techniques.

## Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system that analyzes business ideas for sustainability alignment across multiple regulatory and policy frameworks. The system uses adaptive document retrieval with relevance scoring to provide comprehensive, evidence-based sustainability assessments with full source attribution.

## Key Features

### Advanced AI Capabilities
- **Adaptive Document Retrieval**: Dynamic filtering based on relevance scores (30% threshold)
- **Multi-Vector Store RAG**: Simultaneous querying across multiple knowledge bases
- **Explainable AI**: Full source attribution with relevance statistics
- **Smart Response Generation**: Custom context building with artifact removal
- **LLM-Powered Analysis**: Llama 3.2-1B-Instruct model for fast, accurate responses

### Multi-Framework Analysis
- **UN Sustainable Development Goals (SDGs)**: 17 global goals for sustainable development
- **EU Taxonomy User Guide**: European environmental objectives and criteria
- **Corporate Sustainability Reporting Directive (CSRD)**: EU reporting requirements
- **UN 2030 Agenda**: Global sustainability targets and indicators

### User Interfaces
- **Interactive Streamlit Chat**: Conversational interface with real-time explainability
- **Command Line Interface**: Traditional terminal-based interaction
- **Comprehensive Logging**: Detailed activity tracking and performance monitoring
- **Downloadable Reports**: Detailed analysis reports with full source attribution

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────────┐
│   JSON Files        │    │   Vector Stores     │    │   Adaptive RAG Pipeline │
│                     │    │                     │    │                         │
│ • sdgs.json         │───►│ • sdgs/             │───►│ • Relevance filtering   │
│ • eu_taxonomy.json  │    │ • eu_taxonomy/      │    │ • Custom context build  │
│ • csrd_fact.json    │    │ • csrd_fact/        │    │ • Direct LLM control    │
│ • un_2030.json      │    │ • un_2030/          │    │ • Response cleaning     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────────┐
                           │   Explainable Interface │
                           │                         │
                           │ • Dynamic doc counts    │
                           │ • Relevance statistics  │
                           │ • Source attribution    │
                           │ • Quality metrics       │
                           └─────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (for LLM model)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit langchain-huggingface langchain-community langchain-core
   pip install torch transformers faiss-cpu sentence-transformers
   ```

### Setup

1. **Build Vector Stores**
   ```bash
   python build_faiss_index.py
   ```
   This will automatically process all JSON files and create individual vector stores.

2. **Launch Streamlit Interface**
   ```bash
   streamlit run streamlit_app.py
   ```
   Access at: http://localhost:8501

3. **Or use Command Line Interface**
   ```bash
   python main.py
   ```

## Project Structure

```
Project/
├── README.md                          # This documentation
├── 
├── Data Files
│   ├── sdgs.json                     # UN SDG data
│   ├── eu_taxonomy_user_guide.json   # EU Taxonomy guidelines  
│   ├── csrd_fact_sheet.json          # CSRD requirements
│   └── un_2030_agenda.json           # UN 2030 Agenda
│
├── Core Components
│   ├── build_faiss_index.py          # Auto vector store builder
│   ├── generation_pipeline.py        # Multi-vector RAG pipeline
│   ├── streamlit_app.py              # Web chat with adaptive retrieval
│   ├── causal_model_handler.py       # LLM model handler
│   └── logger.py                     # Logging utility
│
├── Interfaces & Tools
│   ├── main.py                       # CLI interface
│   ├── retrieval_eval.py             # Evaluation tools (standalone)
│   └── test_adaptive_retrieval.py    # Testing utility
│
├── Generated Vector Stores
│   ├── sdgs/                         # SDG vector store
│   ├── eu_taxonomy_user_guide/       # EU Taxonomy vectors
│   ├── csrd_fact_sheet/              # CSRD vectors  
│   └── un_2030_agenda/               # UN 2030 vectors
│
└── Logs/
    ├── faiss_build.log               # Index building logs
    ├── generation.log                # Analysis logs
    └── streamlit_app.log             # UI interaction logs
```

## Advanced Features

### Adaptive Document Retrieval

The system uses advanced retrieval techniques instead of fixed document counts:

- **Relevance Scoring**: Converts FAISS distances to similarity scores (0-100%)
- **Dynamic Filtering**: Only includes documents above 30% relevance threshold
- **Smart Statistics**: Shows relevance rates and quality metrics per knowledge base
- **Variable Document Counts**: Each knowledge base contributes different numbers of documents based on actual relevance

### Explainable AI Interface

- **Source Selection Methodology**: Shows how documents were selected
- **Retrieval Statistics**: Displays relevant vs. total searched documents  
- **Knowledge Base Breakdown**: Individual stats per framework
- **Quality Indicators**: Relevance rates and similarity scores
- **Interactive Source Explorer**: Expandable document previews with metadata

### Enhanced Response Generation

- **Custom Context Building**: Uses only the most relevant documents
- **Direct Model Control**: Bypasses standard pipeline for better control
- **Artifact Removal**: Cleans "[json_content]" and similar noise
- **Response Optimization**: Removes repetition and improves readability

## Usage Examples

### Streamlit Chat Interface

1. **Launch**: `streamlit run streamlit_app.py`
2. **Explore Examples**: Click preset company ideas or type your own
3. **Get Analysis**: Press Enter for comprehensive sustainability analysis
4. **View Sources**: Expand "Sources and Explainability" section
5. **Download Reports**: Get detailed reports with full source attribution

**Sample Analysis Output:**
```
Sources Found: 18 (instead of fixed 24)
Knowledge Bases: 4
Total Searched: 40  
Relevance Rate: 45%

UN SDGs (5 docs, 78% avg relevance)
EU Taxonomy (7 docs, 65% avg relevance)  
CSRD (3 docs, 72% avg relevance)
UN 2030 Agenda (3 docs, 81% avg relevance)
```

### Command Line Interface

```bash
python main.py
# Interactive chat for quick analysis
# Type 'quit' or 'exit' to stop
```

### Example Company Ideas

- "Solar panel manufacturing with AI-powered efficiency optimization"
- "Vertical farming startup using renewable energy and water recycling"
- "Electric vehicle charging network powered by renewable energy"  
- "Sustainable packaging company using biodegradable materials"
- "Carbon capture technology for reducing industrial emissions"

## Technical Configuration

### Model Settings
- **Default Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Relevance Threshold**: 30% (configurable)
- **Max Search Documents**: 10 per source (filtered to relevant ones)
- **Max Response Tokens**: 768
- **Temperature**: 0.2 (for consistent, focused responses)

### Customization Examples

**Adjust Relevance Threshold:**
```python
# In streamlit_app.py
retrieved_docs, stats = adaptive_document_retrieval(
    pipeline, company_idea, relevance_threshold=0.4  # Higher threshold = fewer, more relevant docs
)
```

**Use Different Models:**
```python
pipeline = MultiVectorStoreRAGPipeline(
    llm_model_name="meta-llama/Llama-3.2-3B-Instruct",  # Larger model
    quantize=True,  # Enable quantization for memory efficiency
)
```

**Select Specific Knowledge Bases:**
```python
pipeline = MultiVectorStoreRAGPipeline(
    index_dirs=["sdgs", "eu_taxonomy_user_guide"],  # Only SDGs and EU Taxonomy
)
```

## Performance & Quality

### Performance Metrics
- **Loading Time**: ~10-15 seconds (1B model)
- **Analysis Time**: ~3-8 seconds per query (including adaptive retrieval)
- **Memory Usage**: ~4-6GB RAM
- **Relevance Rate**: Typically 30-60% (adaptive filtering)

### Quality Improvements
- **Dynamic Document Selection**: No more fixed "6, 6, 6, 6" counts
- **Clean Responses**: Removed "[json_content]" artifacts
- **Source Transparency**: Users see exactly which sources informed analysis
- **Quality Metrics**: Real-time feedback on retrieval effectiveness

## Troubleshooting

### Common Issues

**Adaptive Retrieval Not Working:**
```bash
# Test the adaptive retrieval system
python test_adaptive_retrieval.py
```

**Model Loading Errors:**
```bash
# Download models manually if needed
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
"
```

**Vector Store Issues:**
```bash
# Rebuild all vector stores
python build_faiss_index.py
```

**Response Quality Issues:**
- Check `Logs/generation.log` for generation details
- Verify vector stores exist in subdirectories
- Test with simpler company ideas first

### Debug Mode

Enable detailed logging by checking the `Logs/` directory:
- `streamlit_app.log`: UI interactions and retrieval stats
- `generation.log`: Model generation and context building
- `faiss_build.log`: Vector store creation details

## Development & Testing

### Testing Adaptive Retrieval

```bash
python test_adaptive_retrieval.py
```

This will test:
- Document retrieval with relevance scoring
- Statistics calculation
- Context building with filtered documents

### Adding New Knowledge Bases

1. **Add JSON file** to project root directory
2. **Run builder**: `python build_faiss_index.py` (auto-detects new files)
3. **Update friendly names** in `generation_pipeline.py`:
   ```python
   friendly_names = {
       "your_new_file": "Your Friendly Name",
       # ... existing names
   }
   ```
4. **Test integration** with `python test_adaptive_retrieval.py`

### Evaluation Tools

The project includes standalone evaluation capabilities:
- `retrieval_eval.py`: Precision@k and Recall@k metrics
- Can be used independently for retrieval quality assessment
- Supports custom evaluation datasets

## Dependencies

```bash
# Core requirements
streamlit>=1.28.0
langchain-huggingface>=0.1.0  
langchain-community>=0.2.0
langchain-core>=0.2.0
torch>=2.0.0
transformers>=4.35.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2

# Additional utilities
pathlib  # For path handling
json     # For data processing
os       # For file operations
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/adaptive-improvements`
3. **Test changes**: Use `test_adaptive_retrieval.py`
4. **Update documentation**
5. **Submit pull request**

### Code Quality Guidelines

- **Explainability First**: All new features should enhance transparency
- **Performance Awareness**: Consider memory and speed implications
- **Documentation**: Update README for user-facing changes
- **Testing**: Verify with test scripts before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain**: RAG framework and document processing
- **Hugging Face**: Transformers and model ecosystem  
- **FAISS**: Efficient similarity search with scoring
- **Streamlit**: Interactive web application framework
- **Meta**: Llama 3.2 language models
- **UN**: Sustainable Development Goals framework
- **European Commission**: EU Taxonomy and CSRD guidelines

## Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](../../issues) 
- **Documentation**: See inline code comments and logs
- **Testing**: Use provided test scripts for debugging

---

**Built for Explainable Sustainable Business Innovation**

*This tool combines cutting-edge AI with transparency to help entrepreneurs and businesses align their ideas with global sustainability frameworks, promoting responsible innovation with full source attribution and quality metrics.*