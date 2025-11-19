import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda


class ModAgent:
    def __init__(
        self,
        toxicity_model_name=None,
        causal_model_name=None,
        model_provider="hf",             # "hf" | "ollama" | "openai"
        classifier=True,                 # True = use classifier, False = use causal LLM only
        standards=None
    ):
        """
        PARAMETERS:
            toxicity_model_name: path/name of a HF classification model
            causal_model_name: name of instruct model for reasoning
            model_provider: "hf", "ollama", or "openai"
            classifier: if False, skip classifier entirely and rely on causal LLM
        """
        self.model_provider = model_provider
        self.classifier_enabled = classifier

        # --- Load moderation standards ---
        self.standards = standards or "Be civil, respectful, and avoid harassment."

        # --- Load classifier (optional) ---
        if self.classifier_enabled:
            if toxicity_model_name is None:
                toxicity_model_name = 's-nlp/roberta_toxicity_classifier'

            print("Loading toxicity classifier:", toxicity_model_name)
            self.tox_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
            self.tox_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)
        else:
            self.tox_model = None

        # --- Load LLM for reasoning steps ---
        if causal_model_name:
            print("Loading causal model:", causal_model_name)
            self.llm = self._load_llm(causal_model_name, provider=model_provider)


    # PROVIDER SELECTION
    def _load_llm(self, model_name, provider="hf") -> BaseChatModel:

        # --- OpenAI ---
        if provider == "openai":
            from langchain_community.chat_models import ChatOpenAI
            from langchain_community.embeddings import OpenAIEmbeddings
            return ChatOpenAI(
                model=model_name,
                temperature=0,
            )

        # --- Ollama ---
        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model_name,
                temperature=0,
            )

        # --- HuggingFace local model ---
        if provider == "hf":
            from langchain_huggingface import HuggingFacePipeline
            from langchain_huggingface import HuggingFaceEmbeddings
            from transformers import pipeline
            # Auto-detect causal model
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=256,
            )
            return HuggingFacePipeline(pipeline=pipe)

        raise ValueError(f"Unknown provider: {provider}")


    # CLASSIFICATION
    async def check_toxicity(self, text):
        """
        Returns a toxicity probability between 0 and 1.
        """
        if not self.classifier_enabled:
            return None

        inputs = self.tox_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.tox_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        toxicity = probs[0][1].item()

        print("Toxicity score:", toxicity)
        return toxicity


    # LLM REASONING
    async def judge(self, text):
        """
        Uses the causal model to decide moderation action.
        The LLM considers:
            - Community standards
            - Classifier output (if enabled)
        """

        tox = None
        if self.classifier_enabled:
            tox = await self.check_toxicity(text)

        # Moderation reasoning prompt
        prompt = ChatPromptTemplate.from_template("""
You are a moderation assistant.  
Here are the community standards:

{standards}

Message to evaluate:
"{text}"

Toxicity classifier score (if any): {toxicity}

Decide:
1. Should this message be allowed? (yes/no)
2. If no, what rule does it break?
3. What action should the bot take? (warn / delete / timeout / ban)
4. VERY SHORT explanation.
""")

        chain = prompt | self.llm

        response = chain.invoke({
            "text": text,
            "toxicity": tox,
            "standards": self.standards,
        })

        return response.content
