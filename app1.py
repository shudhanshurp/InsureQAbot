import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.set_page_config(page_title="Insurance Chatbot")

@dataclass
class AgentResponse:
    priority: str = ""
    sentiment: str = ""
    response: str = ""

class ClassifierAgent:
    def __init__(self, llm):
        self.llm = llm
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are a question priority classifier.
            Analyze the question and classify it as:
            - HIGH: Urgent matters needing immediate attention (emergencies, time-sensitive issues)
            - MEDIUM: Important but not urgent questions (policy changes, coverage inquiries)
            - LOW: General information requests
            
            Return ONLY the priority level (HIGH, MEDIUM, or LOW) with no other text."""),
            ("human", "{question}")
        ])
        self.chain = self.template | self.llm | StrOutputParser()
    
    def classify(self, question: str) -> str:
        return self.chain.invoke({"question": question})

class SentimentAgent:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        
        self.template = ChatPromptTemplate.from_messages([
            ("system", """Analyze the sentiment and emotional context of the question.
            Consider factors like urgency, frustration, confusion, or satisfaction.
            Provide a brief sentiment analysis that will help in crafting an appropriate response tone.
            
            Context from documents: {context}
            Return ONLY the sentiment analysis in one short phrase."""),
            ("human", "{question}")
        ])
        
        self.chain = (
            {
                "context": lambda x: self.vectorstore.similarity_search(x["question"], k=2),
                "question": lambda x: x["question"]
            }
            | self.template
            | self.llm
            | StrOutputParser()
        )
    
    def analyze_sentiment(self, question: str) -> str:
        return self.chain.invoke({"question": question})

class ResponseAgent:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        
        self.template = ChatPromptTemplate.from_messages([
            ("system", """You are an insurance expert assistant. Generate a response considering:
            
            Priority Level: {priority}
            Customer Sentiment: {sentiment}
            
            Adjust your tone and urgency based on these factors while using the following context to answer:
            {context}
            
            Provide a direct, helpful response that matches the appropriate tone and urgency level.
            answer should contain three things: 1. Priority level, 2. Sentiment, and 3. reponse.
            If you don't know the answer based on the provided context, just say you don't know but try to be helpful."""),
            ("human", "{question}")
        ])
    
    def generate_response(self, question: str, priority: str, sentiment: str) -> str:
        context = self.vectorstore.similarity_search(question, k=3)
        
        chain = (
            {
                "context": lambda _: context,
                "question": lambda _: question,
                "priority": lambda _: priority,
                "sentiment": lambda _: sentiment
            }
            | self.template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({})

class Orchestrator:
    def __init__(self):
        # Check if PDF file exists
        if not os.path.exists('./data/temp.pdf'):
            st.error("Error: temp.pdf not found. Please make sure it exists in the data directory.")
            st.stop()
            
        # Initialize Google LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Load PDF document
        loader = PyPDFLoader('./data/temp.pdf')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        # Initialize embeddings with fallback options
        try:
            import torch
            torch.manual_seed(0)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-MiniLM-L3-v2",
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            st.warning("Using fallback embedding method")
            
            # Direct sentence-transformers implementation
            class SimpleEmbeddings:
                def __init__(self):
                    self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
                
                def embed_documents(self, texts):
                    return self.model.encode(texts, show_progress_bar=False).tolist()
                
                def embed_query(self, text):
                    return self.model.encode(text).tolist()
            
            self.embeddings = SimpleEmbeddings()
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        # Initialize agents
        self.classifier_agent = ClassifierAgent(self.llm)
        self.sentiment_agent = SentimentAgent(self.llm, self.vectorstore)
        self.response_agent = ResponseAgent(self.llm, self.vectorstore)
        
        # Add conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def process_query(self, question: str) -> AgentResponse:
        # Get priority and sentiment in parallel (simulated)
        priority = self.classifier_agent.classify(question)
        sentiment = self.sentiment_agent.analyze_sentiment(question)
        
        # Generate final response using priority and sentiment
        response = self.response_agent.generate_response(question, priority, sentiment)
        
        return AgentResponse(
            priority=priority,
            sentiment=sentiment,
            response=response
        )

try:
    # Initialize the bot
    if "bot" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.bot = Orchestrator()
    
    bot = st.session_state.bot

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your insurance assistant. How can I help you today?"}]

    # Sidebar configuration
    with st.sidebar:
        st.title('Insurance Chatbot')
        st.write("Smart Insurance Assistant")
        st.write("---")
        if st.checkbox("Show debugging info", value=False):
            st.session_state.show_debug = True
        else:
            st.session_state.show_debug = False

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("debug_info") and st.session_state.show_debug:
                st.write("---")
                st.write(f"🎯 Priority: {message['debug_info'].priority}")
                st.write(f"💭 Sentiment: {message['debug_info'].sentiment}")

    # User-provided prompt
    if input_text := st.chat_input("What would you like to know about insurance?"):
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.write(input_text)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                result = bot.process_query(input_text)
                st.write(result.response)
                if st.session_state.show_debug:
                    st.write("---")
                    st.write(f"🎯 Priority: {result.priority}")
                    st.write(f"💭 Sentiment: {result.sentiment}")
        
        message = {
            "role": "assistant", 
            "content": result.response,
            "debug_info": result
        }
        st.session_state.messages.append(message)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Check your API keys and dependencies.")
