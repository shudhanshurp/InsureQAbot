# Insurance Chatbot with RAG and Agentic Workflow

This project implements an interactive chatbot using Streamlit, designed to answer questions about insurance based on information contained within a provided PDF document. It utilizes a Retrieval-Augmented Generation (RAG) approach combined with an agentic workflow to classify user intent, analyze sentiment, and generate contextually relevant and appropriately toned responses.

## Why This Application?

In customer service scenarios, particularly insurance, understanding the user's *intent* and *emotional state* is crucial for providing effective support. A simple Q&A bot might answer a question factually but fail to address underlying urgency or frustration.

This application addresses this by:

1.  **Grounding Responses:** Using RAG ensures answers are based *specifically* on the provided insurance document (e.g., policy details), minimizing hallucinations and providing accurate information.
2.  **Prioritizing Queries:** The `ClassifierAgent` categorizes questions by urgency (HIGH, MEDIUM, LOW), allowing the system (or potentially a human supervisor) to triage requests effectively.
3.  **Understanding Sentiment:** The `SentimentAgent` analyzes the user's likely emotional state (e.g., frustrated, confused, neutral), enabling the `ResponseAgent` to tailor its tone accordingly.
4.  **Structured Response Generation:** The `ResponseAgent` combines the factual context from the document, the assigned priority, and the detected sentiment to generate a helpful, relevant, and empathetic response.
5.  **Modularity & Control:** The agentic workflow (Classifier -> Sentiment -> Response) breaks the problem down into smaller, manageable steps. This makes the system easier to develop, debug, and potentially extend. It provides more explicit control over the response generation process compared to a single monolithic prompt.

Essentially, this application serves as a proof-of-concept for a more sophisticated, context-aware, and empathetic chatbot for specialized domains like insurance, leveraging the power of LLMs while maintaining factual grounding and user-centric interaction.

## Key Features

*   **PDF Document Q&A:** Answers questions based on the content of `./data/temp.pdf`.
*   **Retrieval-Augmented Generation (RAG):** Uses FAISS vector store and embeddings to find relevant document chunks.
*   **Priority Classification:** An LLM-powered agent classifies user questions into HIGH, MEDIUM, or LOW priority.
*   **Sentiment Analysis:** An LLM-powered agent analyzes the sentiment of the user's question.
*   **Context-Aware Responses:** Generates answers considering document context, priority, and sentiment.
*   **Interactive UI:** Built with Streamlit for easy interaction and chat history display.
*   **Debugging Information:** Option to display the classified priority and sentiment for each response.
*   **Fallback Embeddings:** Includes a fallback mechanism using `sentence-transformers` directly if `torch` setup causes issues with `HuggingFaceEmbeddings`.

## Technology Stack

*   **Language:** Python 3.x
*   **LLM Framework:** Langchain
*   **LLM:** Google Gemini (specifically `gemini-2.0-flash` via `langchain-google-genai`)
*   **UI Framework:** Streamlit
*   **Embeddings:** HuggingFace Embeddings (`paraphrase-MiniLM-L3-v2`) / Sentence Transformers (as fallback)
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
*   **PDF Loading:** PyPDFLoader
*   **Text Splitting:** RecursiveCharacterTextSplitter
*   **Environment Variables:** python-dotenv
*   **Data Structures:** dataclasses

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the necessary libraries:
    ```txt
    streamlit
    langchain
    langchain-community
    langchain-google-genai
    faiss-cpu # or faiss-gpu if you have CUDA setup
    pypdf
    sentence-transformers
    python-dotenv
    torch # Required by HuggingFaceEmbeddings/SentenceTransformers
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `faiss-cpu` is generally easier to install. If you encounter issues, consult the FAISS documentation.)*

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the project's root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
    *Obtain your API key from Google AI Studio.*

5.  **Prepare Data:**
    *   Create a directory named `data` in the project root.
    *   Place your insurance PDF document inside the `data` directory and name it `temp.pdf`.

## Running the Application

Execute the Streamlit application from your terminal:

```bash
streamlit run app1.py
```

The application should open in your web browser.

## Configuration

*   **API Key**: Ensure the `GOOGLE_API_KEY` is correctly set in the `.env` file.
*   **PDF Document**: The application expects  `./data/temp.pdf`. Modify the path in the `Orchestrator` class if your file structure is different.
*   **LLM Model**: The model (`gemini-2.0-flash`) and temperature (`0.7`) can be adjusted in the `Orchestrator`'s `__init__` method.
*   **Embeddings Model**: The `HuggingFaceEmbeddings` model (`paraphrase-MiniLM-L3-v2`) can be changed if desired.
*   **Text Splitter**: Chunk size and overlap can be modified in the `Orchestrator`.
