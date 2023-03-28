import json
import pandas as pd
from nltk.corpus import stopwords

from text_preprocessing.text_cleaner import clean_text


class NullPreprocessor:
    """Represent a preprocessor for texts that do not implement lemmatizers."""

    # pylint: disable=too-few-public-methods

    def process_items(self,  # pylint: disable=too-many-arguments,no-self-use
                      items,
                      language,
                      split,
                      remove_punctuations,
                      remove_special_and_digits,
                      remove_stopwords):
        """Process items by only cleaning them.

        Arguments:
            items: List of articles to process.
                   At least expects: [{'id', 'body', 'title'}]
            language: Name of the language of the items.
            split: If the output should be a list of words.
            remove_punctuations: If to remove punctuations.
            remove_special_and_digits: If to remove special characters
                                       and digits.
            remove_stopwords: If to remove stop words.
        """

        if remove_stopwords:
            extended_stopwords = ["said", "say", "mr", "would"]
            stops = stopwords.words(language)
            # print(stops)
            stops.extend(extended_stopwords)
            stops = set(stops)
            print(stops)
        else:
            stops = None
        articles = pd.read_json(json.dumps(items))

        cleaned = [{'id': int(article['article_id']),
                    'article_title': article['title'],
                    'text': clean_text(article['body'],
                                       remove_punctuations,
                                       remove_special_and_digits,
                                       stops,
                                       split),
                   # 'article_url': article['url']
                   }
                   for _, article in articles.iterrows()]

        return cleaned
