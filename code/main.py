
from summarize_speech import *

if __name__ == '__main__':

    path = '/Users/smuddu/galvanize/capstone/data/Speeches/Obama'
    #path = '/Users/smuddu/galvanize/capstone/data/Speeches/samples'
    #path = '/Users/smuddu/galvanize/talkingpoints/data/Romney'
    #path = '/Users/smuddu/galvanize/talkingpoints/data/simple'

    ''' using URLs '''
    #path = '/Users/smuddu/galvanize/talkingpoints/data/just_links_obama'

    #vocab, doc2topic, topics, model = extract_corpus_topics(path,2)
    extract_corpus_topics(path,15,1,5)

    ''' print top topics '''
    #print_top_topics(topics)

    #extract_speech_excerpts(path, vocab, doc2topic, model, 1, 3)

