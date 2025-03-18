# Text-Summarization
It summarize the text.
import nltk
import heapq
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, summary_length=3):
    # Step 1: Clean the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    
    # Step 2: Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Step 3: Calculate word frequencies
    word_frequencies = {}
    for word in words:
        if word not in stop_words and word.isalpha():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    # Step 4: Score sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
    
    # Step 5: Extract top sentences
    summary_sentences = heapq.nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Example Usage
if __name__ == "__main__":
    text = """Natural Language Processing (NLP) is a field of artificial intelligence that gives machines the ability to read, understand, and derive meaning from human languages. It is used to apply algorithms to identify and extract the natural language rules such that the unstructured language data is converted into a form that machines can understand. The process involves parsing text into its basic components, such as words and sentences, understanding grammar, and interpreting meaning. NLP has numerous applications, including chatbots, machine translation, sentiment analysis, and summarization."""
    
    print("Original Text:\n", text)
    print("\nSummarized Text:\n", summarize_text(text))

