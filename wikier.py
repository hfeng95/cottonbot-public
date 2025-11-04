import spacy
from spacy.cli import download
from langchain_community.retrievers import WikipediaRetriever


class WikiAgent:
    def __init__(self):
        print('Loading WikiAgent')

        try:
            # Attempt to load the model. TODO: other languages?
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Model 'en_core_web_sm' not found. Downloading...")
            # Download the model if it's not found
            download("en_core_web_sm")
            # Load the model after successful download
            self.nlp = spacy.load("en_core_web_sm")

        self.wiki = WikipediaRetriever(top_k_results=1)
        self.cache = {}

    def extract_keywords(self, query: str):
        """Extracts key nouns and named entities, sorted by rarity/importance."""
        # casing workaround. alternatively you can try uncased model or truecasing
        query_combined = query + '\n' + query.title()

        doc = self.nlp(query_combined)

        candidates = [token for token in doc if token.pos_ in ("NOUN", "PROPN")]
        entities = [ent.text for ent in doc.ents]

        # Sort by rarity: lower rank = rarer = more informative
        candidates = sorted(candidates, key=lambda t: t.rank if t.has_vector else 1e9)
        keywords = [t.text for t in candidates]
        
        # Merge with named entities (always keep those)
        combined = list(dict.fromkeys(entities + keywords))  # preserve order, deduplicate
        return combined

    def lookup(self,query):
        if not query in self.cache:
            # remove page/summary label
            full_result = self.wiki.run(query)
            index = full_result.find('Summary: ')
            if index != -1:
                self.cache[query] = full_result[index+9:]
            else:
                self.cache[query] = full_result  
        return self.cache[query]
    
    def lookup_body(self,query,top_k_keywords=1):
        keyword_list = self.extract_keywords(query)[:top_k_keywords]
        results = [self.lookup(k) for k in keyword_list]
        return results




