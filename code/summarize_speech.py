
'''
Talking Points

This tool extract topics of speeches and summarizes them

'''

import nltk
import string
import os
import re

import unidecode
import operator
import pprint

import numpy as np
from collections import defaultdict

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF

from goose import Goose



stemmer = PorterStemmer()
pp = pprint.PrettyPrinter(indent=4)


def grab_link(in_url):
    ''' extract article information using Python Goose '''
    try:
        article = Goose().extract(url=in_url)
        return article
    except ValueError:
        print 'Goose failed to extract article from url'
        return None
    return None


def create_corpus(url_file, raw=False):
    raw_sp2txt = {}
    proc_sp2txt = {}
    speech_titles = {}
    U = open(url_file)
    url_list = [url.strip() for url in U.readlines()]
    for doc_index, url in enumerate(url_list):
        pp.pprint('Grabbing URL: ' + str(url))

        article = grab_link(url)

        title = unidecode.unidecode_expect_nonascii(article.title)
        speech_titles[doc_index] = title

        _raw_input = article.cleaned_text
        text = unidecode.unidecode_expect_nonascii(_raw_input)
        re.sub("[\W\d]", " ", text.lower().strip())
        lowers = text.replace('\n',' ').replace('\r',' ')
        while "  " in lowers:
            lowers = lowers.replace('  ',' ')


        ''' store raw text -- for sentence extraction '''
        raw_sp2txt[doc_index] = lowers

        ''' store no_punctuation for NMF '''
        no_punctuation = lowers.translate(None, string.punctuation)
        proc_sp2txt[doc_index] = no_punctuation

    U.close()
    return proc_sp2txt, raw_sp2txt, speech_titles




def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    return tokens


def parse_speeches(corpus_path):
    raw_sp2txt = {}
    proc_sp2txt = {}
    for subdir, dirs, files in os.walk(corpus_path):
        for each_file in files:
            #pp.pprint("-- processing: {}".format(each_file))
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

    return proc_sp2txt, raw_sp2txt


def get_corpus_topics(tfidf, model, n_topics):
    ''' vocabulary ID to word mapping '''
    id2word = {}
    topics = []
    for k in tfidf.vocabulary_.keys():
        id2word[tfidf.vocabulary_[k]] = k

    for topic_index in xrange(n_topics):
        #pp.pprint("\n-- Top words in topic:")
        topic_importance = dict(zip(id2word.values(),list(model.components_[topic_index])))
        sorted_topic_imp = sorted(topic_importance.items(), key=operator.itemgetter(1),reverse=True)
        #pp.pprint([i[0] for i in sorted_topic_imp[:15]])
        topics.append([i[0] for i in sorted_topic_imp])

    ''' list of all words sorted in descending order of importance for all topics '''
    return topics


def print_top_topics(topics, n_topics=10):
    pp.pprint([i[0:9] for i in topics[:n_topics]])


def get_top_topics(W, n_topics):
    top_topics = []
    for row in W:
        top_topics.append(np.argsort(row)[::-1][:n_topics])

    return top_topics


def extract_corpus_topics(corpus_path, n_corpus_topics, n_doc_topics=1, n_summary_sentences=5):

    ''' Parse contents of speech directory and get dictionaries '''
    proc_speech, raw_speech = parse_speeches(corpus_path)

    #proc_speech, raw_speech, titles = create_corpus(corpus_path)



    ''' TFIDF vectorization and generate vocabularies '''
    corpus_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs_corpus = corpus_tfidf.fit_transform(proc_speech.values())

    #print "Corpus TF vector shape: {}".format(tfs_corpus.shape)

    ''' Get the vocabulary from TF-IDF - for tokenizing in future steps '''
    corpus_vocab = corpus_tfidf.get_feature_names()

    ''' create a NMF model '''
    corpus_model = NMF(n_components=n_corpus_topics, init='random', random_state=0)
    corpusW = corpus_model.fit_transform(tfs_corpus)

    #print "Shape of W (decomposition output) = {}".format(Wcorpus.shape)
    ''' get *all* words for each topic '''
    topics = get_corpus_topics(corpus_tfidf, corpus_model, n_corpus_topics)

    '''
    For each document/speech -- extract the top sentences
    Create a dict of dicts to populate sentences for every speech in the corpus
    '''
    speech_sentences = defaultdict(dict)
    raw_sentences = defaultdict(dict)

    ''' Create a sentence TF-IDF instance using the corpus vocabulary '''
    sentence_tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', vocabulary=corpus_vocab)

    ''' top topics for each document '''
    best_topic_indices = get_top_topics(corpusW, n_doc_topics)

    '''
    (1) iterate over raw speech text and speech_sentences
    (2) get sentence term-frequency vectors based on the vocabulary of the corpus
    (3) check the cosine similarity of every sentences' TF vector with that of the top topics for that document
    '''
    for index,doc in enumerate(raw_speech.iterkeys()):
        print "*"*120
        pp.pprint('Processing: ' + str(doc))
        #pp.pprint('Title: ' + titles[doc])

        doc_blob = TextBlob(raw_speech[doc])
        sentence_count = 0
        for sentence in doc_blob.sentences:
            ''' strip punctuation from the sentence now '''
            sentence_no_punct = str(sentence).translate(None, string.punctuation)
            speech_sentences[doc][sentence_count] = sentence_no_punct
            raw_sentences[doc][sentence_count] = sentence
            sentence_count += 1

        speech_tfs = sentence_tfidf.fit_transform(speech_sentences[doc].values()).todense()

        ''' iterate over the speech's most-relevant topics - and get cosine similarity '''
        top_topics_of_doc = best_topic_indices[index]

        for topic_index in top_topics_of_doc:

            pp.pprint('Top Topic: ' + str(topic_index))
            pp.pprint('Top Topic Words: ' + str(topics[topic_index][:10]))
            print ""

            topic_vector = corpus_model.components_[topic_index]
            sentence_similarity = {}
            for s_index, s_tf in enumerate(speech_tfs):
                ''' calcuating the cosine similarity with this sort of reshape op -- to get rid of a sklearn warning '''
                sentence_similarity[s_index] = cosine_similarity(s_tf,topic_vector.reshape((1,-1)))[0][0]

            ''' sort the sentence_similarity and pull the indices of top sentences '''
            top_n_sentences = [i[0] for i in sorted(sentence_similarity.items(), key=operator.itemgetter(1), reverse=True)[:n_summary_sentences]]
            for i in top_n_sentences:
                pp.pprint(str(raw_sentences[doc][i]))
                print ""


    return
