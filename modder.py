import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

default_toxicity_model_name = 's-nlp/roberta_toxicity_classifier'
actions = ['kick','ban','mute','delete']

class ModAgent:
    def __init__(self,toxicity_model_name=None,standards=None):
        if toxicity_model_name is None:
            toxicity_model_name = default_toxicity_model_name
        print('Loading toxicity model',toxicity_model_name)
        self.tox_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_name)
        self.tox_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_name)
        if standards:
            self.standards = standards

    async def check_toxicity(self,text):
        inputs = self.tox_tokenizer.encode(text,return_tensors='pt')
        with torch.no_grad():
            logits = self.tox_model(inputs).logits
        probs = torch.softmax(logits, dim=-1)
        toxicity = probs[0][1].item()
        print('Toxicity score:',toxicity)
        return toxicity
    
    async def judge(self,text):
        # TODO: causal model to determine action
        pass