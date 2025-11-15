# Multi-Vector Store RAG Sustainability Analysis Tool

A comprehensive AI-powered sustainability analysis system that evaluates company ideas against multiple authoritative frameworks including UN SDGs, EU Taxonomy, CSRD, and UN 2030 Agenda.

## Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system that analyzes business ideas for sustainability alignment across multiple regulatory and policy frameworks. The system uses multiple vector stores to provide comprehensive, evidence-based sustainability assessments.

## Features

### Multi-Framework Analysis
- **UN Sustainable Development Goals (SDGs)**: 17 global goals for sustainable development
- **EU Taxonomy**: European environmental objectives and criteria
- **Corporate Sustainability Reporting Directive (CSRD)**: EU reporting requirements
- **UN 2030 Agenda**: Global sustainability targets and indicators

### Advanced AI Capabilities
- **Multi-Vector Store RAG**: Simultaneous querying across multiple knowledge bases
- **Intelligent Document Retrieval**: Context-aware similarity search
- **LLM-Powered Analysis**: Llama 3.2 1B Instruct model for fast, accurate responses
- **Smart Context Building**: Automatic organization by knowledge source

### User Interfaces
- **Interactive Streamlit Chat**: Conversational interface with real-time responses
- **Command Line Interface**: Traditional terminal-based interaction
- **Comprehensive Logging**: Detailed activity tracking and performance monitoring

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   JSON Files        │    │   Vector Stores     │    │   RAG Pipeline      │
│                     │    │                     │    │                     │
│ • sdgs.json         │───►│ • sdgs/             │───►│ • Multi-retrieval   │
│ • eu_taxonomy.json  │    │ • eu_taxonomy/      │    │ • Context building  │
│ • csrd_fact.json    │    │ • csrd_fact/        │    │ • LLM generation    │
│ • un_2030.json      │    │ • un_2030/          │    │ • Response cleaning │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   User Interface    │
                           │                     │
                           │ • Streamlit Chat    │
                           │ • CLI Interface     │
                           │ • Download Reports  │
                           └─────────────────────┘
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
   pip install -r requirements.txt
   ```

3. **Install required packages**
   ```bash
   pip install langchain-huggingface langchain-community langchain-core
   pip install faiss-cpu torch transformers streamlit
   ```

### Setup

1. **Build Vector Stores**
   ```bash
   python3 build_faiss_index.py
   ```
   This will automatically process all JSON files and create individual vector stores.

2. **Launch Streamlit Interface**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Or use Command Line Interface**
   ```bash
   python3 main.py
   ```

## Project Structure

```
Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── 
├── Data Files
│   ├── sdgs.json                     # UN SDG data
│   ├── eu_taxonomy_user_guide.json   # EU Taxonomy guidelines
│   ├── csrd_fact_sheet.json          # CSRD requirements
│   └── un_2030_agenda.json           # UN 2030 Agenda
│
├── Core Components
│   ├── build_faiss_index.py          # Vector store builder
│   ├── generation_pipeline.py        # Multi-vector RAG pipeline
│   ├── causal_model_handler.py       # LLM model handler
│   └── logger.py                     # Logging utility
│
├── User Interfaces
│   ├── streamlit_app.py              # Web chat interface
│   └── main.py                       # CLI interface
│
├── Generated Vector Stores
│   ├── sdgs/                         # SDG vector store
│   ├── eu_taxonomy_user_guide/       # EU Taxonomy vectors
│   ├── csrd_fact_sheet/              # CSRD vectors
│   └── un_2030_agenda/               # UN 2030 vectors
│
└── Logs
    ├── faiss_build.log               # Index building logs
    ├── generation.log                # Analysis logs
    └── streamlit_app.log             # UI interaction logs
```

## Usage Examples

### Streamlit Chat Interface

1. Launch the app: `streamlit run streamlit_app.py`
2. Click example ideas or type your own company concept
3. Press Enter to get comprehensive sustainability analysis
4. Download reports in Markdown format

### Command Line Interface

```bash
python3 main.py
# Type company ideas and get analysis
# Type 'quit' or 'exit' to stop
```

### Example Company Ideas

- "Solar panel manufacturing with AI-powered efficiency optimization"
- "Vertical farming startup using renewable energy and water recycling"
- "Electric vehicle charging network powered by renewable energy"
- "Sustainable packaging company using biodegradable materials"

## Configuration

### Model Settings
- **Default Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Documents per Source**: 6 (retrievs 24 total documents)
- **Max Response Tokens**: 768
- **Temperature**: 0.2 (for consistent, focused responses)

### Customization

#### Using Different Models
```python
pipeline = MultiVectorStoreRAGPipeline(
    llm_model_name="meta-llama/Llama-3.2-3B-Instruct",  # Larger model
    quantize=True,  # Enable quantization for memory efficiency
)
```

#### Selecting Specific Knowledge Bases
```python
pipeline = MultiVectorStoreRAGPipeline(
    index_dirs=["sdgs", "eu_taxonomy_user_guide"],  # Only SDGs and EU Taxonomy
)
```

## Requirements

Create a `requirements.txt` file:

```
streamlit>=1.28.0
langchain-huggingface>=0.1.0
langchain-community>=0.2.0
langchain-core>=0.2.0
torch>=2.0.0
transformers>=4.35.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
pathlib>=1.0.1
```

## Technical Details

### Vector Store Creation
- **Automatic Format Detection**: Distinguishes SDG format from generic JSON
- **Intelligent Chunking**: Groups content for optimal retrieval
- **Metadata Preservation**: Maintains source attribution and document types

### RAG Pipeline
- **Multi-Source Retrieval**: Queries all knowledge bases simultaneously
- **Context Organization**: Groups information by framework
- **Response Cleaning**: Removes repetition and formats output

### Performance Optimizations
- **Model Caching**: Streamlit caches loaded models
- **Efficient Embedding**: CPU-optimized sentence transformers
- **Smart Tokenization**: Automatic truncation and padding

## Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Download models manually if needed
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct'); AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"
```

**Vector Store Not Found**
```bash
# Rebuild vector stores
python3 build_faiss_index.py
```

**Memory Issues**
- Use quantized models: Set `quantize=True`
- Reduce batch sizes in analysis
- Close other applications to free RAM

**Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear
```

## Performance Metrics

### Model Performance
- **Loading Time**: ~10-15 seconds (1B model)
- **Analysis Time**: ~2-5 seconds per query
- **Memory Usage**: ~4-6GB RAM
- **Accuracy**: Comprehensive multi-framework coverage

### Scalability
- **Concurrent Users**: 5-10 (depending on hardware)
- **Document Capacity**: 100k+ documents per vector store
- **Knowledge Bases**: Unlimited (memory permitting)

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Add your changes**
4. **Update tests and documentation**
5. **Submit pull request**

### Adding New Knowledge Bases

1. **Add JSON file** to project directory
2. **Run vector store builder**: `python3 build_faiss_index.py`
3. **Update knowledge base names** in `generation_pipeline.py`
4. **Test integration**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain**: RAG framework and document processing
- **Hugging Face**: Transformers and model ecosystem
- **FAISS**: Efficient similarity search
- **Streamlit**: Web application framework
- **Meta**: Llama 3.2 language models
- **UN**: Sustainable Development Goals framework
- **European Commission**: EU Taxonomy and CSRD guidelines

## Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](../../issues)
- **Documentation**: See inline code comments
- **Logs**: Check `Logs/` directory for detailed information

---

**Built for sustainable business innovation**

*This tool helps entrepreneurs and businesses align their ideas with global sustainability frameworks, promoting responsible innovation and environmental stewardship.*