import streamlit as st
import os
from pathlib import Path
import time
from typing import List, Dict

from generation_pipeline import MultiVectorStoreRAGPipeline
from logger import Logger


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False


def get_logger() -> Logger:
    """
    Create a Logger instance for Streamlit app.
    """
    logs_dir = Path("Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = logs_dir / "streamlit_app.log"

    logger = Logger(
        name="StreamlitApp",
        log_file_needed=True,
        log_file=str(log_file_path),
        level="DEV",
    )
    return logger


@st.cache_resource
def load_pipeline():
    """
    Load the Multi-Vector Store RAG pipeline with caching for better performance.
    """
    logger = get_logger()
    logger.info("Loading Multi-Vector Store RAG Pipeline for Streamlit...")
    
    try:
        pipeline = MultiVectorStoreRAGPipeline(
            llm_model_name="meta-llama/Llama-3.2-1B-Instruct",
            quantize=False,
        )
        logger.info("Pipeline loaded successfully")
        return pipeline, logger
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        st.error(f"Failed to load pipeline: {str(e)}")
        return None, logger


def display_knowledge_bases(pipeline, logger):
    """
    Display information about loaded knowledge bases in the sidebar.
    """
    with st.sidebar:
        st.subheader("📚 Knowledge Bases")
        if hasattr(pipeline, 'vectorstores') and hasattr(pipeline, 'knowledge_base_names'):
            for source_name, friendly_name in pipeline.knowledge_base_names.items():
                if source_name in pipeline.vectorstores:
                    st.success(f"✅ {friendly_name}")
                    logger.debug(f"Knowledge base displayed: {friendly_name}")
        else:
            st.warning("No knowledge bases loaded")
            
        st.subheader("ℹ️ About")
        st.write("""
        This app analyzes company ideas against multiple sustainability frameworks:
        - **UN SDGs**: Sustainable Development Goals
        - **EU Taxonomy**: Environmental objectives
        - **CSRD**: Corporate reporting requirements  
        - **UN 2030 Agenda**: Global sustainability targets
        """)
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            logger.info("Chat history cleared by user")
            st.rerun()


def display_analysis_results(analysis_text: str, logger):
    """
    Display the analysis results with proper formatting.
    """
    if not analysis_text:
        st.warning("No analysis generated")
        return
    
    # Split the analysis into sections based on numbered headings or clear breaks
    sections = analysis_text.split('\n\n')
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if section starts with a number (like "1. Overall Assessment")
        if section.startswith(('1.', '2.', '3.', '4.')):
            # Extract heading and content
            lines = section.split('\n')
            heading = lines[0]
            content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            # Display with appropriate formatting
            if "Overall Assessment" in heading:
                st.subheader("🎯 " + heading)
            elif "Key Alignments" in heading:
                st.subheader("🎯 " + heading)
            elif "Regulatory" in heading:
                st.subheader("⚖️ " + heading)
            elif "Recommendations" in heading:
                st.subheader("💡 " + heading)
            else:
                st.subheader("📋 " + heading)
                
            if content:
                st.write(content)
        else:
            # Regular content
            st.write(section)
    
    logger.debug("Analysis results displayed successfully")


def main():
    # Configure page
    st.set_page_config(
        page_title="Sustainability Analysis Tool",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize logger
    logger = get_logger()
    logger.info("Streamlit app started")
    
    # Header
    st.title("🌱 Sustainability Analysis Tool")
    st.subheader("Analyze your company ideas against multiple sustainability frameworks")
    
    # Load pipeline
    with st.spinner("Loading AI models and knowledge bases..."):
        pipeline, logger = load_pipeline()
    
    if pipeline is None:
        st.error("❌ Failed to load the analysis pipeline. Please check the logs.")
        logger.error("Pipeline loading failed, stopping app")
        return
    
    # Display knowledge bases in sidebar
    display_knowledge_bases(pipeline, logger)
    
    # Main interface
    st.markdown("---")
    
    # Chat interface
    st.subheader("💬 Chat with the Sustainability Advisor")
    
    # Example ideas in expander
    with st.expander("💭 Need inspiration? Click here for example ideas"):
        example_buttons = st.columns(2)
        
        with example_buttons[0]:
            if st.button("🔋 Solar + AI Manufacturing", use_container_width=True):
                st.session_state.user_input = "Solar panel manufacturing company with AI-powered efficiency optimization"
            if st.button("🌱 Vertical Farming", use_container_width=True):
                st.session_state.user_input = "Vertical farming startup using renewable energy and water recycling systems"
            if st.button("⚡ EV Charging Network", use_container_width=True):
                st.session_state.user_input = "Electric vehicle charging network powered entirely by renewable energy"
            if st.button("📦 Sustainable Packaging", use_container_width=True):
                st.session_state.user_input = "Sustainable packaging company using biodegradable materials from agricultural waste"
                
        with example_buttons[1]:
            if st.button("🌍 Carbon Capture Tech", use_container_width=True):
                st.session_state.user_input = "Carbon capture technology for reducing industrial emissions"
            if st.button("♻️ Circular Economy Platform", use_container_width=True):
                st.session_state.user_input = "Digital platform connecting businesses for waste material exchange and reuse"
            if st.button("🏗️ Ocean Plastic Buildings", use_container_width=True):
                st.session_state.user_input = "Green building materials manufactured from recycled ocean plastic"
            if st.button("🔌 Smart Grid Technology", use_container_width=True):
                st.session_state.user_input = "AI-powered smart grid technology for optimizing renewable energy distribution"
    
    # Chat messages container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(f"🏢 **Company Idea:** {message['content']}")
                else:
                    st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input(
        "Describe your company idea and press Enter...",
        key="chat_input"
    )
    
    # Handle user input from chat or example buttons
    if user_input or ("user_input" in st.session_state and st.session_state.user_input):
        # Use example input if available, otherwise use chat input
        if "user_input" in st.session_state and st.session_state.user_input:
            company_idea = st.session_state.user_input
            st.session_state.user_input = ""  # Clear after use
        else:
            company_idea = user_input
            
        if company_idea and company_idea.strip():
            logger.info(f"New user message: {company_idea[:100]}...")
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": company_idea})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(f"🏢 **Company Idea:** {company_idea}")
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your idea against sustainability frameworks..."):
                    try:
                        start_time = time.time()
                        
                        # Generate analysis with fixed parameters
                        analysis = pipeline.generate(
                            idea=company_idea,
                            k_per_source=6,  # Fixed value
                            max_new_tokens=768,  # Fixed value
                            temperature=0.2  # Fixed value
                        )
                        
                        end_time = time.time()
                        analysis_time = end_time - start_time
                        
                        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
                        
                        # Format the response with timing
                        response = f"## 📊 Sustainability Analysis\n\n{analysis}\n\n---\n*Analysis completed in {analysis_time:.1f} seconds*"
                        
                        # Display the analysis
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Offer download for the latest analysis
                        st.download_button(
                            label="📥 Download This Analysis",
                            data=f"# Sustainability Analysis Report\n\n**Company Idea:**\n{company_idea}\n\n{analysis}",
                            file_name=f"analysis_{int(time.time())}.md",
                            mime="text/markdown",
                            key=f"download_{len(st.session_state.messages)}"
                        )
                        
                    except Exception as e:
                        error_msg = f"❌ Analysis failed: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Analysis failed: {str(e)}")
                        
                        # Add error to chat history
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Auto-scroll to bottom (rerun to show new messages)
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌱 Built with Streamlit • Powered by Multi-Vector Store RAG • AI Sustainability Advisor</p>
        <p><small>💬 Chat interface for analyzing company ideas against UN SDGs, EU Taxonomy, CSRD, and UN 2030 Agenda</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()