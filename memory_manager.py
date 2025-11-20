from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory, VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
import os

DEFAULT_MODEL_TYPE = 'no-chain' # openai/ollama/hf/no-chain. no-chain uses homebrew implementation. TODO: implement frontend


class CottonMemory:
    def __init__(self, model_name=None, model_type=DEFAULT_MODEL_TYPE,memory_dir="memory_data"):
        self.memory_dir = memory_dir
        self.model_type = model_type

        os.makedirs(memory_dir, exist_ok=True)

        # --- Model & Embeddings selection ---
        if self.model_type=='hf':
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_huggingface import HuggingFacePipeline
            from langchain_huggingface import HuggingFaceEmbeddings

            if model_name==None:
                model_name = "LiquidAI/LFM2-350M"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto"
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        elif self.model_type=='ollama':
            from langchain_community.chat_models import ChatOllama
            from langchain_huggingface import HuggingFaceEmbeddings

            if model_name==None:
                model_name = "mistral:latest"

            self.llm = ChatOllama(model=model_name, temperature=0.7)
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        elif self.model_type=='openai':
            from langchain_community.chat_models import ChatOpenAI
            from langchain_community.embeddings import OpenAIEmbeddings

            if model_name==None:
                model_name = 'gpt-4o-mini'

            self.llm = ChatOpenAI(model=model_name, temperature=0.7)
            self.embeddings = OpenAIEmbeddings()
        elif self.model_type=='no-chain':
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            from langchain_huggingface import HuggingFacePipeline
            from langchain_huggingface import HuggingFaceEmbeddings

            if model_name==None:
                model_name = 'eliori/dialogue_summarization-finetuned'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto"
            )
            pipe = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.conversation_summary = ''

        # Load or initialize FAISS vector memory
        faiss_found = self.load()
        if not faiss_found:
            self.vectorstore = FAISS.from_texts(["cottonbot awakens anew."], self.embeddings)

        # Memory layers
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.summary = ConversationSummaryMemory(llm=self.llm, memory_key="summary")

    def save(self,path=None):
        """Persist memory to disk."""
        if path==None:
            path = self.memory_dir
        os.makedirs(path, exist_ok=True)
        self.vectorstore.save_local(os.path.join(path, "long_term.faiss"))
    
    def load(self,path=None):
        """Load long term FAISS memory from disk."""
        if path==None:
            path = self.memory_dir
        faiss_path = os.path.join(path, "long_term.faiss")
        if os.path.exists(faiss_path):
            self.vectorstore = FAISS.load_local(
                faiss_path, self.embeddings, allow_dangerous_deserialization=True
            )
            return True
        return False

    def save_context(self, user_input, bot_output, user_name='User',bot_name='cottonbot',num_rounds=3):
        """Store new chat turns."""
        self.memory.save_context({"input": user_input}, {"output": bot_output})
        if self.model_type=='no-chain':
            recent_messages = '\n'.join(self.get_recent_conversation(user_name=user_name,bot_name=bot_name,num_rounds=num_rounds))
            combined_text = f"{self.conversation_summary}\n{recent_messages}"
            self.conversation_summary = self.llm.invoke(combined_text, max_new_tokens=150, min_length=30, do_sample=False)
            print('Summary:',self.conversation_summary)
        else:
            self.summary.save_context({"input": user_input}, {"output": bot_output})
        
        # Add conversation to vectorstore for long-term memory
        conversation_text = f"{user_name}: {user_input}\n{bot_name}: {bot_output}"
        self.vectorstore.add_texts([conversation_text])

    def get_context(self):
        """Retrieve memory context for prompting."""
        return self.memory.load_memory_variables({})
    
    def get_buffer(self):
        if self.model_type=='no-chain':
            return self.conversation_summary
        return self.summary.buffer
    
    def get_recent_conversation(self, user_name='User',bot_name='cottonbot',num_rounds=3,return_separated=False):
        """Return the most recent n rounds of conversation (user + bot pairs)."""
        messages = self.memory.chat_memory.messages[-2*num_rounds:]
        pairs = []
        for i in range(0, len(messages), 2):
            user_msg = messages[i].content if i < len(messages) else ""
            bot_msg = messages[i+1].content if i+1 < len(messages) else ""
            if return_separated:
                pairs.append(((user_name,user_msg),(bot_name,bot_msg)))
            else:
                pairs.append(f"{user_name}: {user_msg}\n{bot_name}: {bot_msg}")
        return pairs

    async def reflect(self):
        """Generate reflective summaries to improve future recall."""
        prompt = f"""
            Review your memory and summarize any new facts, relationships, or ideas that may be useful later.
            {self.get_context()}
            """
        reflection = self.llm.predict(prompt)
        self.retriever.save_context({"input": "self_reflection"}, {"output": reflection})
        return reflection
