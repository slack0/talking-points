
'''
Speech Transcript processing / cleaning to feed the NLP pipeline
'''

import nltk
import string
import os
import re

import unidecode
import operator
import pprint

from collections import defaultdict

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    return tokens


def parse_speeches(corpus_path, raw=False):
    raw_sp2txt = {}
    proc_sp2txt = {}
    for subdir, dirs, files in os.walk(corpus_path):
        for each_file in files:
            pp.pprint("-- processing: {}".format(each_file))
            file_path = subdir + os.path.sep + each_file
            fhandle = open(file_path, 'r')
            _raw_input = fhandle.read()
            text = unidecode.unidecode_expect_nonascii(_raw_input)
            re.sub("[\W\d]", " ", text.lower().strip())
            lowers = text.replace('\n',' ').replace('\r',' ')
            while "  " in lowers:
                lowers = lowers.replace('  ',' ')

            ''' store raw text -- for sentence extraction '''
            raw_sp2txt[each_file] = lowers

            ''' store no_punctuation for NMF '''
            no_punctuation = lowers.translate(None, string.punctuation)
            proc_sp2txt[each_file] = no_punctuation

    if (raw == True):
        return raw_sp2txt
    else:
        return proc_sp2txt


def get_corpus_topics(tfidf, model, n_topics):
    ''' vocabulary ID to word mapping '''
    id2word = {}
    topics = []
    for k in tfidf.vocabulary_.keys():
        id2word[tfidf.vocabulary_[k]] = k

    for topic_index in xrange(n_topics):
        pp.pprint("\n-- Top words in topic:")
        topic_importance = dict(zip(id2word.values(),list(model.components_[topic_index])))
        sorted_topic_imp = sorted(topic_importance.items(), key=operator.itemgetter(1),reverse=True)
        pp.pprint([i[0] for i in sorted_topic_imp[:15]])
        topics.append([i[0] for i in sorted_topic_imp])

    ''' list of all words sorted in descending order of importance for all topics '''
    return topics


def print_top_topics(topics, n_topics=10):
    pp.pprint([i[0] for i in topics[:n_topics]])


def extract_corpus_topics(corpus_path, num_topics):

    ''' Parse contents of speech directory and get dictionaries '''
    proc_speech = parse_speeches(corpus_path)


    ''' TFIDF vectorization and generate vocabularies '''
    corpus_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs_corpus = corpus_tfidf.fit_transform(proc_speech.values())

    #print "Corpus TF vector shape: {}".format(tfs_corpus.shape)


    ''' Get the vocabulary from TF-IDF - for tokenizing in future steps '''
    corpus_vocab = corpus_tfidf.get_feature_names()


    ''' create a NMF model '''
    corpus_model = NMF(n_components=num_topics, init='random', random_state=0)
    corpusW = corpus_model.fit_transform(tfs_corpus)

    #print "Shape of W (decomposition output) = {}".format(Wcorpus.shape)
    ''' get *all* words for each topic '''
    topics = get_corpus_topics(corpus_tfidf, corpus_model, num_topics)

    ''' print top topics '''
    print_top_topics(topics)

    return

def extract_speech_excerpts(corpus_path, corpus_vocab, W):

    ''' Parse contents of speech directory and get dictionaries '''
    raw_speech = parse_speeches(corpus_path, raw=True)

    '''
    For each document/speech -- extract the top sentences
    Create a dict of dicts to populate sentences for every speech in the corpus
    '''
    speech_sentences = defaultdict(dict)

    ''' Create a sentence TF-IDF instance using the corpus vocabulary '''
    sentence_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', vocabulary=corpus_vocab)


    ''' iterate over raw speech text and speech_sentences '''
    for doc in raw_speech.iterkeys():
        #pp.pprint('Processing: ' + str(doc))

        doc_blob = TextBlob(raw_speech[doc])
        sentence_count = 0
        for sentence in doc_blob.sentences:
            ''' strip punctuation from the sentence now '''
            sentence_without_punctuation = str(sentence).translate(None, string.punctuation)
            speech_sentences[doc][sentence_count] = sentence_without_punctuation
            sentence_count += 1

    ''' map top topics to each document '''

        speech_tfs = sentence_tfidf.fit_transform(speech_hash[doc].values())
        print "\nSpeech TF vector shape: {}".format(speech_tfs.shape)


        ''' check the cosine similarity of each sentence against topic tfidfs '''
        #distances = cosine_similarity(speech_tfs,corpus_tfs)
        #print "Shape of cosine distance vector: {}".format(distances.shape)

        #pp.pprint(distances)





if __name__ == '__main__':

    stemmer = PorterStemmer()
    pp = pprint.PrettyPrinter(indent=2)

    path = '/Users/smuddu/galvanize/capstone/data/Speeches/Obama'
    #path = '/Users/smuddu/galvanize/capstone/data/Speeches/samples'

    extract_corpus_topics(path,4)
