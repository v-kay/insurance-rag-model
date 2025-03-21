import time
import re
import streamlit as st
import boto3
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM as Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage

# Import UI styles
from styles import STYLES, BANNER, TYPING

class AWSConnector:
    """Manages AWS service connections and configurations."""
    
    def __init__(self, region_name='us-east-1'):
        """Initialize AWS service clients.
        
        Args:
            region_name (str): AWS region to connect to
        """
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.s3 = boto3.client(service_name="s3")
        
    def get_bedrock_llm(self, model_id="amazon.nova-micro-v1:0", max_gen_len=512):
        """Initialize and return a Bedrock LLM instance.
        
        Args:
            model_id (str): The Bedrock model identifier
            max_gen_len (int): Maximum generation length
            
        Returns:
            Bedrock: Initialized LLM instance
        """
        return Bedrock(
            model_id=model_id, 
            client=self.bedrock,
            model_kwargs={'max_gen_len': max_gen_len}
        )
    
    def get_bedrock_embeddings(self, model_id="amazon.titan-embed-text-v1"):
        """Initialize and return Bedrock embeddings.
        
        Args:
            model_id (str): The Bedrock embeddings model identifier
            
        Returns:
            BedrockEmbeddings: Initialized embeddings instance
        """
        return BedrockEmbeddings(model_id=model_id, client=self.bedrock)


class VectorStoreManager:
    """Manages vector storage operations including downloading and loading vectors."""
    
    def __init__(self, bucket_name, file_path, s3_client, embeddings):
        """Initialize the vector store manager.
        
        Args:
            bucket_name (str): S3 bucket name containing vectors
            file_path (str): Local path to store downloaded vectors
            s3_client: Boto3 S3 client instance
            embeddings: Embeddings instance for vector operations
        """
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.s3_client = s3_client
        self.embeddings = embeddings
        
        # Ensure the file path exists
        Path(self.file_path).mkdir(parents=True, exist_ok=True)
        
    def download_vectors(self, policy_number):
        """Download vector files from S3 for a specific policy.
        
        Args:
            policy_number (str): Policy identifier for vector retrieval
        """
        s3_vector_faiss_key = f'vectors/policydoc/{policy_number}/policydoc_faiss.faiss'
        s3_vector_pkl_key = f'vectors/policydoc/{policy_number}/policydoc_pkl.pkl'
        
        self.s3_client.download_file(
            Bucket=self.bucket_name, 
            Key=s3_vector_faiss_key, 
            Filename=f"{self.file_path}/my_faiss.faiss"
        )
        self.s3_client.download_file(
            Bucket=self.bucket_name, 
            Key=s3_vector_pkl_key, 
            Filename=f"{self.file_path}/my_faiss.pkl"
        )
        
    def load_faiss_index(self, llm, prompt_template):
        """Load FAISS index and create a retrieval chain.
        
        Args:
            llm: Language model instance
            prompt_template: Template for query/context formatting
            
        Returns:
            chain: Configured retrieval chain
        """
        # Load the FAISS index from local storage
        faiss_index = FAISS.load_local(
            index_name="my_faiss", 
            folder_path=self.file_path, 
            embeddings=self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Configure the retrieval components
        retriever = faiss_index.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_template)
        
        # Create and return the complete chain
        return create_retrieval_chain(retriever_chain, document_chain)


class ChatBot:
    """Main chatbot class handling user interaction, policy validation, and responses."""
    
    def __init__(self):
        """Initialize the chatbot with required components and configurations."""
        # Constants
        self.BUCKET_NAME = "rag-bot-source"
        self.FILE_PATH = "/tmp"
        
        # Initialize AWS connector
        self.aws = AWSConnector(region_name='us-east-1')
        
        # Configure LLM and embeddings
        self.llm = self.aws.get_bedrock_llm()
        self.embeddings = self.aws.get_bedrock_embeddings()
        
        # Setup prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "input"],
            template="""Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer.
2. If you find the answer, write the answer in a detailed way without references.
{context}
Question: {input}
Helpful Answer:"""
        )
        
        # Initialize vector store manager
        self.vector_store = VectorStoreManager(
            self.BUCKET_NAME,
            self.FILE_PATH,
            self.aws.s3,
            self.embeddings
        )
        
        # Configure Streamlit UI
        self._setup_ui()
        
        # Initialize session state
        self._initialize_session_state()
        
    def _setup_ui(self):
        """Configure Streamlit UI components."""
        st.set_page_config(page_title="First Bot", page_icon="🤖")
        st.markdown(STYLES, unsafe_allow_html=True)
        st.markdown(BANNER, unsafe_allow_html=True)
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Initialize chat history if not already present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm a chat bot, your virtual assistant"),
            ]
            time.sleep(2)
            st.session_state.chat_history.append(
                AIMessage(content="Please enter your policy number to get started")
            )
            st.session_state.policy_id_validated = False
            
        if "awaiting_response" not in st.session_state:
            st.session_state.awaiting_response = False
            
    def validate_policy_id(self, policy_id):
        """Validate if the provided policy ID matches the expected format.
        
        Args:
            policy_id (str): Policy ID to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return bool(re.match(r'^AU\d{4}$', policy_id))
        
    def get_response(self, query, chain):
        """Get streaming response from the retrieval chain.
        
        Args:
            query (str): User query
            chain: Retrieval chain to use
            
        Returns:
            Generator: Stream of response chunks
        """
        return chain.stream({"input": query})
    
    def get_streamed_response(self, prompt, chain):
        """Aggregate streaming response chunks into a complete response.
        
        Args:
            prompt (str): User prompt/query
            chain: Retrieval chain to use
            
        Returns:
            str: Complete response text
        """
        chunks = []
        for chunk in self.get_response(prompt, chain):
            if "answer" in chunk:
                chunks.append(chunk["answer"])
        return ''.join(chunks)
    
    def display_chat_history(self):
        """Display all messages in the chat history."""
        for message in st.session_state.chat_history:
            # Set avatar URLs based on message type
            avatar_url = (
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4h4UlldDaq69Pd6QHlzSVB8yAYH73Gpn5Qkn5R2fYS10XfhpKlr86Ci8-HjyX0ft9Ivw&usqp=CAU" 
                if isinstance(message, HumanMessage) 
                else "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQG1JzTwTj8b1jzq2zKlBbLEf3i-rOLwnmZqQ&usqp=CAU"
            )
            
            # Set bubble class based on message type
            bubble_class = "user-bubble" if isinstance(message, HumanMessage) else "assistant-bubble"
            
            # Display message with avatar
            st.markdown(f"""
                <div class="chat-bubble {bubble_class}">
                    <img src="{avatar_url}" class="chat-avatar">
                    {message.content}
                </div>
            """, unsafe_allow_html=True)
    
    def handle_user_input(self):
        """Process user input based on policy validation state."""
        # Set appropriate input prompt based on policy validation state
        if not st.session_state.policy_id_validated:
            prompt_ = st.chat_input("Enter your policy number(e.g., AU1234789):")
        else:
            prompt_ = st.chat_input("Type your message here")
            
        # Handle user message if provided
        if prompt_:
            st.session_state.chat_history.append(HumanMessage(content=prompt_))
            st.session_state.awaiting_response = True
            st.experimental_rerun()
            
    def process_response(self):
        """Process and generate responses based on user input and policy state."""
        if st.session_state.awaiting_response:
            # Show typing indicator
            typing_indicator = st.empty()
            with typing_indicator:
                st.markdown(TYPING, unsafe_allow_html=True)
                
            # Get the latest user message
            user_input = st.session_state.chat_history[-1].content
            
            # Handle policy validation or query response
            if not st.session_state.policy_id_validated:
                if self.validate_policy_id(user_input):
                    # Valid policy: download vectors and set up chain
                    self.vector_store.download_vectors(user_input)
                    st.session_state.chain = self.vector_store.load_faiss_index(
                        self.llm, 
                        self.prompt_template
                    )
                    response = "Policy number validated successfully. How can I help about your policy today?"
                    st.session_state.policy_id_validated = True
                else:
                    # Invalid policy number
                    response = "Incorrect policy number. Please enter a valid policy"
            else:
                # Process query with retrieval chain
                response = self.get_streamed_response(user_input, st.session_state.chain)
                
            # Add response to chat history
            st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.awaiting_response = False
            
            # Clear typing indicator and refresh UI
            typing_indicator.empty()
            st.experimental_rerun()
            
    def run(self):
        """Main execution method to run the chatbot."""
        # Display existing chat history
        self.display_chat_history()
        
        # Handle new user input
        self.handle_user_input()
        
        # Process responses if awaiting
        self.process_response()


# Application entry point
if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()
