"""Prepare a corpus for subsequent use.

Use the `corpus_pipeline` method to create a 'cleaned' corpus.
Or run this file with python. Example:
  python clean_corpus.py -i INPUT -o OUTPUT -l LANGUAGE \
         -p PREPROCESSOR_NAME \
         --split \
         --remove_punctuations \
         --remove_special \
         --remove_stopwords

Change INPUT, OUTPUT, LANGUAGE and PREPROCESSOR_NAME to respective values
If PREPROCESSOR_NAME not given, then NullPreprocessor will be used.

Input .json file is expected to have, at least, the following format:
  [{'id': int', body': string}]

Output .json file will have this format:
  {
    'language': string,
    'language_code': string,
    'corpus': [{'article_id': int,
                'article_title': str,
                'text': string or [string]}]
  }

"""

import json
from argparse import ArgumentParser

# from text_preprocessing.lemmatag.preprocessor import LemmaTagPreprocessor
from text_preprocessing.null_preprocessor import NullPreprocessor
# from text_preprocessing.spacy_lemmy.preprocessor import SpacyLemmyPreprocessor
from text_preprocessing.nltk_preprocessor import NltkPreprocessor
from typing import Optional, List, Dict


PARSER = ArgumentParser()
PARSER.add_argument("-l", "--language",
                    help="language of corpus")
PARSER.add_argument("-i", "--input",
                    help="filename of raw corpus .json file", metavar="FILE")
PARSER.add_argument("-o", "--output",
                    help="filename where to save cleaned corpus .json file",
                    metavar="FILE")
PARSER.add_argument("-p", "--preprocessor",
                    help="Preprocessor name.",
                    nargs='?',
                    const=None)

PARSER.add_argument("--split", action='store_true')
PARSER.add_argument("--remove_punctuations", action='store_true')
PARSER.add_argument("--remove_special", action='store_true')
PARSER.add_argument("--remove_stopwords", action='store_true')

LANGUAGE_CODES = {
    'swedish': 'sv',
    'english': 'en'
}

PREPROCESSORS = {None: NullPreprocessor(),
                #  'spacy': SpacyLemmyPreprocessor(),
                #  'lemmatag': LemmaTagPreprocessor(),
                 'nltk': NltkPreprocessor(), }


def corpus_pipeline(language: str,  # pylint: disable=too-many-arguments
                    raw_data_path: str,
                    output_path: str,
                    preprocessor_name: Optional[str] = None,
                    split: bool = False,
                    remove_punctuations: bool = True,
                    remove_special_and_digits: bool = True,
                    remove_stopwords: bool = True):
    """Load given corpus, 'cleans' it and saves result as .json.

    Arguements:
        language: language of corpus
        raw_data_path: path to raw data .json file
        output_path: where to save results
        split: boolean if final output should be split into list of words.
           If false, will be kept as a single string
        remove_punctuations: boolean if punctuations should be removed
        remove_special_and_digits: boolean if special characters like
           `!@#$%^` and digits should be removed
        remove_stopwords: boolean if stopwords should be removed
    """

    with open(raw_data_path, 'r') as file:
        articles = json.load(file)

    clean_corpus = make_corpus(articles,
                               language,
                               preprocessor_name,
                               split,
                               remove_punctuations,
                               remove_special_and_digits,
                               remove_stopwords)

    save_corpus(clean_corpus, output_path)


def make_corpus(raw_data: List[Dict],  # pylint: disable=too-many-arguments
                language: str,
                preprocessor_name: Optional[str] = None,
                split: bool = False,
                remove_punctuations: bool = False,
                remove_special_and_digits: bool = False,
                remove_stopwords: bool = False):
    """Create and return a refined corpus.

    Arguments:
        raw_data: raw-data as Json. Atleast expects: [{id, 'body'}]
        language: language of corpus
        split: boolean if final output should be split into list of words.
           If false, will be kept as a single string
        remove_punctuations: boolean if punctuations should be removed
        remove_special_and_digits: boolean if special characters like
           `!@#$%^` and digits should be removed
        remove_stopwords: boolean if stopwords should be removed
    """
    language = language.lower()
    if language not in LANGUAGE_CODES.keys():
        raise ValueError(f"Language '{language}' not supported. "
                         "Available languages are: {LANGUAGE_CODES.keys()}")
    preprocessor = PREPROCESSORS[preprocessor_name].process_items
    cleaned_corpus = preprocessor(items=raw_data,
                                  language=language,
                                  split=split,
                                  remove_punctuations=remove_punctuations,
                                  remove_special_and_digits=remove_special_and_digits,  # noqa E501
                                  remove_stopwords=remove_stopwords)

    return {'language': language,
            'language_code': LANGUAGE_CODES[language],
            'corpus': cleaned_corpus}


def save_corpus(clean_corpus, filename):
    """Write the given corpus to disk as a .json file."""
    with open(filename, 'w', encoding='utf8') as file:
        json.dump(clean_corpus, file, ensure_ascii=False)


def main():
    args = PARSER.parse_args()
    corpus_pipeline(args.language,
                    args.input,
                    args.output,
                    args.preprocessor,
                    args.split,
                    args.remove_punctuations,
                    args.remove_special,
                    args.remove_stopwords)


if __name__ == '__main__':
    main()
