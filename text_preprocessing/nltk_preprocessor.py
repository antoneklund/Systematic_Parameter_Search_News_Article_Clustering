"""Implement NLTK lemmatizer along with NLTK POS tagger for English.

The NLTK package is added in the requirements.txt
If you still have problem with any of the NLTK methods/data missing,
please check: http://www.nltk.org/data.html
"""

import datetime
import logging
import logging.config as config
from typing import Dict
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from settings.base import LOGGING_CONF
from text_preprocessing.text_cleaner import clean_text

config.fileConfig(LOGGING_CONF)
LOGGER = logging.getLogger(__name__)


class NltkLemmatizerEn:
    """Implement NLTK based lemmatizer for English.

    Use nltk's pos_tag method to get POS tagging of the words
    and use the POS taggs for WordNetLemmatizer to get better
    accuracy in lemmatizing.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def wordnet_pos_tag(self, treebank_tag):  # pylint: disable=no-self-use
        """Convert Treebank tag format to WordNet tag format.

        This concept is taken from:
        https://stackoverflow.com/questions/15586721/
        wordnet-lemmatization-and-pos-tagging-in-python
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        if treebank_tag.startswith('V'):
            return wordnet.VERB
        if treebank_tag.startswith('N'):
            return wordnet.NOUN
        if treebank_tag.startswith('R'):
            return wordnet.ADV

        return ''
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def do_lemmatize(self, text):
        # return [self.lemmatizer.lemmatize(wordtuple[0],
        #                                   self.wordnet_pos_tag(wordtuple[1]))
        #         if self.wordnet_pos_tag(wordtuple[1])
        #         else wordtuple[0]
        #         for wordtuple in nltk.pos_tag(text)]
        return [self.lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]


class NltkStemmer:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def stem_sentence(self, sentence):
        token_words = word_tokenize(sentence)
        stemmed_sentence = []
        for word in token_words:
            stemmed_sentence.append(self.stemmer.stem(word))
            stemmed_sentence.append(" ")
        return "".join(stemmed_sentence)

class NltkPreprocessor:
    # pylint: disable=too-few-public-methods

    def process_items(self,  # pylint: disable=too-many-arguments,no-self-use
                      items: List[Dict],
                      language: str = 'english',
                      split: bool = False,
                      remove_punctuations: bool = True,
                      remove_special_and_digits: bool = True,
                      remove_stopwords: bool = True):
        """Process items by only cleaning them.

        Arguments:
            items: Articles to process. At least expects: [id, 'title', 'body']
            language: Name of the language of the items.
            split: If the output should be a list of words.
            remove_punctuations: If to remove punctuations.
            remove_special_and_digits: If to remove special characters
                                       and digits.
            remove_stopwords: If to remove stop words.
        """
        LOGGER.info("Preprocessing. Might take a while.")
        start = datetime.datetime.now()
        lemmatizer = NltkLemmatizerEn()
        stemmer = NltkStemmer()

        if remove_stopwords:
            extended_stopwords = ["said", "say", "mr", "would", "ve", "just", "yes", "no", "don"]
            stops = stopwords.words(language)
            stops.extend(extended_stopwords)
            stops = set(stops)
        else: 
            None
        cleaned = []

        for _, article in pd.DataFrame.from_dict(items).iterrows():
            text = clean_text(article['text'],
                              remove_punctuations,
                              remove_special_and_digits,
                              stops,
                              False)

            # text = [text] #When lemmatizing, it wants list of strings
            cleaned_text = lemmatizer.do_lemmatize(text) #take the string in the list when lemmatizing
            word_body = ""
            for w in cleaned_text:
                word_body = word_body + w + " "
            cleaned_text = word_body
            cleaned.append({'article_id': int(article['article_id']),
                            'text': cleaned_text})

        LOGGER.info('Text preprocessing took %s (H:M:S.MS)',
                    (datetime.datetime.now() - start))
        return cleaned
