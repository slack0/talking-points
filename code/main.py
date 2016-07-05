
from summarize_speech import *
#from utils import *

if __name__ == '__main__':

    static_html_corpus = {
            'obama': '../data/speech_corpus/obama_raw_html',
            'romney': '../data/speech_corpus/romney_raw_html',
            'test': '../data/speech_corpus/simple_html'
    }

    static_text_corpus = {
            'obama' : '../data/Obama',
            'romney': '../data/Romney',
            'test': '../data/tests/simple'
    }

    curated_url_lists = {
            'obama': '../data/obama.links',
            'romney': '../data/romney.links',
            'trump': '../data/trump.links',
            'test': '../data/tests/test.links'
    }
    path = curated_url_lists['trump']
    vocab, doc2topic, topics = extract_corpus_topics(path,10)

    ''' print top topics '''
    print_top_topics(topics)


