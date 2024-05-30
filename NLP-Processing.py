# nlp_processing.py

"""
NLP Processing Module for Voice-Activated Virtual Assistant

This module contains functions for processing and understanding natural language
to extract meaning and intent from user queries.

Techniques Used:
- Tokenization
- Part-of-Speech Tagging
- Named Entity Recognition
- Sentiment Analysis

Libraries/Tools:
- NLTK
- spaCy
- Transformers (BERT, GPT)
"""

import spacy
from transformers import pipeline

class NLPProcessing:
    def __init__(self):
        """
        Initialize the NLPProcessing class.
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = pipeline('sentiment-analysis')

    def tokenize(self, text):
        """
        Tokenize the input text.
        
        :param text: str, input text
        :return: list, tokens
        """
        doc = self.nlp(text)
        return [token.text for token in doc]

    def pos_tagging(self, text):
        """
        Perform part-of-speech tagging on the input text.
        
        :param text: str, input text
        :return: list, tokens with their part-of-speech tags
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def named_entity_recognition(self, text):
        """
        Perform named entity recognition on the input text.
        
        :param text: str, input text
        :return: list, named entities with their labels
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def sentiment_analysis(self, text):
        """
        Perform sentiment analysis on the input text.
        
        :param text: str, input text
        :return: dict, sentiment analysis result
        """
        return self.sentiment_analyzer(text)[0]

if __name__ == "__main__":
    # Example usage
    text = "Apple is looking at buying U.K. startup for $1 billion. The company's stock price surged after the news."

    nlp_processor = NLPProcessing()
    
    # Tokenization
    tokens = nlp_processor.tokenize(text)
    print("Tokens:", tokens)
    
    # Part-of-Speech Tagging
    pos_tags = nlp_processor.pos_tagging(text)
    print("Part-of-Speech Tags:", pos_tags)
    
    # Named Entity Recognition
    entities = nlp_processor.named_entity_recognition(text)
    print("Named Entities:", entities)
    
    # Sentiment Analysis
    sentiment = nlp_processor.sentiment_analysis(text)
    print("Sentiment Analysis:", sentiment)
