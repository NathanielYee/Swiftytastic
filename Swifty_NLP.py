'''
Nathaniel Yee and Dhruv Rokkam
Tuesday Nov 21st 2023

'''

# Import necessary libraries and modules
from collections import Counter
import pdfplumber
import re
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# Define a custom exception for text analysis errors
class TextAnalysisException(Exception):
    def __init__(self, filename, error_message):
        super().__init__(f"Error processing file {filename}: {error_message}")
        self.data = {}

    # Define a method to handle non-PDF files
    def non_pdf(self):
        return f"The file entered is not a pdf document"

# Define a default text parser function
def default_parser(text):
    # Preprocess text: convert to lowercase, remove non-alphanumeric characters, and remove Table of Contents
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'Table of Contents.*?\n', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered = [token for token in tokens if token.lower() not in stop_words]
    # Count word frequencies
    word_freq = Counter(filtered)

    # Perform sentiment analysis for each sentence
    analyze = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    sentiment_scores = [analyze.polarity_scores(sentence) for sentence in sentences]

    return {"word_freq": word_freq, "sentiment": sentiment_scores}

# Define a class for text analysis
class TextAnalyzer:
    def __init__(self):
        self.data = {}  # Stores data for multiple files

    # Load text from a PDF file, apply custom parser if provided
    def load_text(self, filename, label="", custom_parser=None):
        if not label:
            label = filename

        try:
            # Use pdfplumber to extract text from PDF
            with pdfplumber.open(filename) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            # Apply custom parser or default parser to the text
            if custom_parser:
                self.data[label] = custom_parser(text)
            else:
                self.data[label] = default_parser(text)
        except Exception as e:
            # Raise an exception for any errors during text analysis
            raise TextAnalysisException(filename, f"PDF Syntax Error: {str(e)}")

    # Create a Sankey diagram for word frequency analysis
    def wordcount_sankey(self, word_list=None, k=10):
        labels = []
        source = []
        target = []
        values = []

        # If no word list is provided, use the most common words
        if not word_list:
            all_words = Counter()
            for text_data in self.data.values():
                all_words.update(text_data["word_freq"])
            word_list = [word for word, count in all_words.most_common(k)]

        labels.extend(self.data.keys())
        word_index = len(self.data)
        for word in word_list:
            labels.append(word)
            for text_label, text_data in self.data.items():
                if word in text_data["word_freq"]:
                    source.append(list(self.data.keys()).index(text_label))
                    target.append(word_index)
                    values.append(text_data["word_freq"][word])
            word_index += 1

        # Create and display a Sankey diagram using Plotly
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                      label=labels, color="blue"),
            link=dict(source=source, target=target, value=values))])
        fig.update_layout(title_text="Text to Word Sankey Diagram", font_size=10)
        fig.show()

    # Create a visualization of word clouds for each document
    def your_second_visualization(self, subplot_rows=2, subplot_cols=5):
        fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 10))
        axes = axes.flatten()

        for i, (text_label, text_data) in enumerate(self.data.items()):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                text_data["word_freq"])
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(text_label)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig('second.png')
        plt.show()

    # Create a bar chart for average sentiment scores across documents
    def your_third_visualization(self):
        plt.figure(figsize=(10, 6))

        labels = []
        avg_sentiment_scores = []

        # Calculate average sentiment scores for each document
        for label, data in self.data.items():
            compound_scores = [score['compound'] for score in data['sentiment']]
            avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
            labels.append(label)
            avg_sentiment_scores.append(avg_compound)

        # Create and display a bar chart
        plt.bar(labels, avg_sentiment_scores, color='blue')
        plt.xlabel('Documents')
        plt.ylabel('Average Sentiment Score')
        plt.title('Comparative Sentiment Analysis Across Documents')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to accommodate label names
        plt.savefig('third.png')
        plt.show()
def main():
    # build + load model
    analyzer = TextAnalyzer()
    analyzer.load_text("oursong.pdf", "Our Song- 'Taylor Swift' ")
    analyzer.load_text('youbelongwithme.pdf', "You Belong With Me- 'Fearless'")
    analyzer.load_text('speaknow.pdf', 'Mean- "Speak Now"')
    analyzer.load_text('alltoowell.pdf', 'All Too Well- "Red"')
    analyzer.load_text('blankspace.pdf', 'Blank Space- "1989"')
    analyzer.load_text('delicate.pdf', 'Delicate- "Reputation"')
    analyzer.load_text('cruelsummer.pdf', 'Cruel Summer- "Lover"')
    analyzer.load_text('exile.pdf', 'Exile- "Folklore"')
    analyzer.load_text('ivy.pdf', 'Ivy- "Evermore"')
    analyzer.load_text('antihero.pdf', 'Anti-Hero- "Midnights"')
    # create plots
    analyzer.wordcount_sankey()
    analyzer.your_second_visualization(subplot_rows=2, subplot_cols=5)
    analyzer.your_third_visualization()


if __name__ == '__main__':
    main()