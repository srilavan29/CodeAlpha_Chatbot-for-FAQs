import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class ChatbotLogic:
    def __init__(self, faq_file='faqs.json'):
        self.faq_file = faq_file
        self.faqs = self.load_faqs()
        self.questions = [faq['question'] for faq in self.faqs]
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def load_faqs(self):
        with open(self.faq_file, 'r') as f:
            return json.load(f)

    def preprocess_text(self, text):
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words]
        
        return filtered_tokens

    def get_response(self, user_question):
        # Transform user question
        user_tfidf = self.vectorizer.transform([user_question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
        
        # Get the index of the most similar question
        max_similarity_index = similarities.argmax()
        max_similarity_score = similarities[0, max_similarity_index]
        
        # Threshold for matching
        if max_similarity_score > 0.15:
            return self.faqs[max_similarity_index]['answer']
        else:
            return "I'm sorry, I don't have an answer for that. Can you please rephrase your question?"

if __name__ == "__main__":
    # Test the logic
    bot = ChatbotLogic()
    test_q = "What is Zylith?"
    print(f"Q: {test_q}")
    print(f"A: {bot.get_response(test_q)}")
    
    test_q2 = "Is it free?"
    print(f"Q: {test_q2}")
    print(f"A: {bot.get_response(test_q2)}")
    
    test_q3 = "How can I contact you?"
    print(f"Q: {test_q3}")
    print(f"A: {bot.get_response(test_q3)}")
