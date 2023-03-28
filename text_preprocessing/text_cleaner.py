"""Clean a piece of text depending on conditions provided.

A text is anything that is a string. Could one or multiple sentences
and one or multiple paragraphs.

For now, cleaning is done by performing any of the following operaitons:
  - Removing punctuations
  - Removing special characters, like !@#%^, and digits
  - Removing stopwords (stopword library taken from nltk-package)
More cleaning actions may be added in the future.

The 'text' of output data can also optionally be formatted into token
form, i.e. a list of words. The default format of 'text' is a single
string.
"""

import re
from typing import Union, List


def clean_text(text: str,
               remove_punctuations: bool,
               remove_special_and_digits: bool,
               stop_words: Union[None, set, List],
               split: bool,
               prepare_bert: bool = False,
               reuters: bool = False) -> Union[str, List[str]]:
    """Clean and return text based on given conditions.

    Arguments:
        text - Raw text.
        remove_punctuations - Boolean if punctuations should be removed.
        remove_special_and_digits - Boolean if special characters like
                                    `!@#$%^` and digits should be removed.
        stop_words - List of stopwords to be removed.
                     If None, no stopwords will be removed.
        split - Boolean if text output should be split into list of words.
                If false, will be kept as a single string.
    """
    # Convert to lowercase
    text = text.lower()

    # Prepare the reuters dataset
    if reuters:
        text = re.sub("Calif\.,", "", text) # Remove Calif., to help City and Date removal
        text = re.sub("\(Editing by.+", "", text) # Remove everything after (Editing by ...)
        text = re.sub("\(Reporting by.+", "", text) # Remove everything after (Reporting by ...)
        text = re.sub("(\S+ +)?\S+\, +\S+ [0-9][0-9]? +\(Reuters\)", "", text) # Remove CITY and Date
        text = re.sub("By \S+ \S+", "", text) # Remove By First Last (can't solve more names)
        text = re.sub("\[n[A-Za-z0-9]+]", "", text) # Remove links [n.....]
        text = re.sub("\[\S+]", "", text) # Remove [TEXT] 
        text = re.sub("\<\S+\>", "", text) # Remove <TEXT>
        text = re.sub("Keywords: .+", "", text)

    # Preserve abbreviations: Turns abbreviations such as U.S.A into USA
    # So that we don't lose abbreviated words while removing punctuations
    text = re.sub(r'(?<!\w)([A-Za-z0-9])\.', r'\1', text)

    # remove tabs
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # Prepares the corpus for bert tokenizer
    # Keeps punctuation
    if prepare_bert:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)  # add combine "..." to "."

    # Remove punctuations preserving compound words
    # Not work properly, keep: word-
    if remove_punctuations:
        text = re.sub(r'\.', ' ', text)  
        text = re.sub(r'\s+', ' ', text)

    if remove_special_and_digits:
        text = re.sub(r'[\'’‘]s', '', text)  # remove "*'s" which results in " s "
        text = re.sub(r'[\'’‘]ll', '', text)
        text = re.sub(r'[\'’‘]re', '', text)

        # special_characters = r"""[!@#$€%“”"’‘^–*&\-',;~@*+:<>(){}=|`´?\[\]^\\]"""
        # text = re.sub(special_characters, ' ', text)
        text = re.sub(r'[^a-öA-ÖæøåÆØÅ\.]', ' ', text)

        text = re.sub(r'\d+', ' ', text)

    if stop_words:
        # Find the stop word and replace with a space
        for stop in stop_words:
            stop = r'\b%s\b' % stop
            text = re.sub(stop, ' ', text)
    
    if split:
        return text.split()

    return text
