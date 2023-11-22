"""
filename: textastic.py
description: An extensible reusable library for text analysis and comparison
"""

from collections import defaultdict, Counter
import random as rnd
import matplotlib.pyplot as plt


class Textastic:

    def __init__(self):
        # string  --> {filename/label --> statistics}
        # "wordcounts" --> {"A": wc_A, "B": wc_B, ....}
        self.data = defaultdict(dict)

    def _save_results(self, label, results):
        for k, v in results.items():
            self.data[k][label] = v

    @staticmethod
    def _default_parser(filename):
        """ DEMONSTRATION ONLY:
        Extracting word counts and number of words
        as a random number.
        Replace with a real parser that processes
        your input file fully.  (Remove punctuation,
        convert to lowercase, etc.)   """

        results = {
            'wordcount': Counter("to be or not to be".split(" ")),
            'numwords': rnd.randrange(10, 50)
        }
        return results

    def load_text(self, filename, label=None, parser=None):
        """ Registers a text document with the framework
        Extracts and stores data to be used in later
        visualizations. """

        if parser is None:
            results = Textastic._default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        # store the results of processing one file
        # in the internal state (data)
        self._save_results(label, results)

    def compare_num_words(self):
        """ A DEMONSTRATION OF A CUSTOM VISUALIZATION
        A trivially simple barchart comparing number
        of words in each registered text file. """

        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.show()

    def viz2(self):
        pass

    def viz3(self):
        pass

