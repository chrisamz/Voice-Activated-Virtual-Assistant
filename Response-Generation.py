# response_generation.py

"""
Response Generation Module for Voice-Activated Virtual Assistant

This module contains functions for generating coherent and contextually relevant
responses to user queries using various models.

Techniques Used:
- Sequence-to-Sequence Models
- Attention Mechanisms
- Transformer Models

Libraries/Tools:
- Transformers (GPT-2, T5, BERT)
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

class ResponseGeneration:
    def __init__(self):
        """
        Initialize the ResponseGeneration class.
        """
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def generate_response_gpt2(self, prompt, max_length=50):
        """
        Generate a response using the GPT-2 model.
        
        :param prompt: str, input prompt
        :param max_length: int, maximum length of the generated response
        :return: str, response
        """
        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=self.gpt2_tokenizer.eos_token_id)
        return self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_response_t5(self, prompt, max_length=50):
        """
        Generate a response using the T5 model.
        
        :param prompt: str, input prompt
        :param max_length: int, maximum length of the generated response
        :return: str, response
        """
        inputs = self.t5_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.t5_model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=self.t5_tokenizer.eos_token_id)
        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    response_generator = ResponseGeneration()
    user_query = "Tell me a joke."

    # Generate response using GPT-2
    response_gpt2 = response_generator.generate_response_gpt2(user_query)
    print("GPT-2 Response:", response_gpt2)

    # Generate response using T5
    response_t5 = response_generator.generate_response_t5(user_query)
    print("T5 Response:", response_t5)
