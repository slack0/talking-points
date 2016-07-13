
# Talking Points

NLP-based summarization tool for politcal speeches and interviews.

## Introduction

Speech summarization is a necessity in many domains. It is needed in
cases where the audience needs to get a gist of a speech or an article
in a way that preserves the intent and the meaning of the speech. This
problem belongs to a class of problems under [Automatic
summarization](https://en.wikipedia.org/wiki/Automatic_summarization).


Summary generation is difficult. It is context dependent.  The objective
and the metrics of summarization process themselves are difficult to
define in some situations. Extracting information that is representative
of an entire discussion/topic/speech is nuanced, as the speaker may
convey meaning implicitly while referring to a past or a future
discussion.

The main idea of summarization is to create a representative summary or
an abstract which contains the information of the entire set. This is
achieved by finding the most informative sentences in the
document/speech.


## Summary Generation from Political Speeches

### Motivation

The primary motivation for this work is to develop insights and build a
profile of a speaker based on their speech. For a speaker/leader in
public domain, who is well studied and discussed in the media, it is
relatively easy to develop insights. But developing insights into
someone who is not in the limelight is difficult. Speeches, interviews,
and conversations in general reveal what people believe in and where
they stand on issues. Speech patterns such as vocabulary usage,
sentiments expressed and topics discussed are some of the main
attributes that pop out directly from speeches. Using these primary
attributes, we explore summary generation.

### Methodology

In this work, we consider the problem of summary extraction from
documents/speeches based on the knowledge of what topics the speech is
about.

The main intuition here is that the inference about the topic is
valuable in evaluating which parts of a document or a speech are
relevant to it.  Using this intuition, this tool provides a summary of
the most important topics of speeches based on speaking style (word
usage) of a specific person (i.e., from a speech corpus). Based on topic
extraction and sentence/document vectorization, the tool extracts most
relevant / important sentences from a speech by ranking topic similarity
to the sentence similarity.

Results show that this technique outperforms similar summarization
techniques that rely only on sentence similarity.

### Topic Extraction

The first step in summary generation from speeches is to identify the
topic(s) associated with any given speech. Topic extraction is effective
if it is done on a corpus of speeches as opposed to analyzing each
individual speech. To extract the topics of a corpus of speeches (a.k.a
corpus), we perform vectorization of the speeches using TF-IDF
(term-frequency inverse document frequency). The TF-IDF vectorization
provides vectorized word representations with vocabulary set to the
entire corpus. Using the vectorized representation of the corpus, we
then perform non-negative matrix factorization (NMF) to bring out the
latent topics of the corpus. NMF provides the mapping between 
speeches-to-topics and topics-to-vocabulary.

The speeches-to-topic mapping reveals interesting details about the
distribution of topics related to each speech within the corpus. The
figure below shows the distribution of topics related to six speeches
from Obama. The specificity of a speech is clearly evident from this
visual. Obviously, some speeches are concerned with specific topics,
while others discuss a combination of topics. It also revals that the
vocabulary used by the speaker (Obama in this case) was specific enough
to be captured into distinct topics.

![alt
text](https://raw.githubusercontent.com/slack0/talking-points/master/data/topic_distribution.png "Distribution of topics for speeches")

### Topic Mapping to Speeches

Every speech/document is a combination of topics. For instance, a press
conference given by Obama may cover ongoing wars, the military, economy,
health care, congress, income inequality and education. In contrast, a
speech to a business forum may just be about the state of economy. The
topic distributions in each of these two cases will be different. 

Speeches can be considered as distributions over topics. And topics as
distributions over vocabulary. The mapping between (speeches, topics)
and (topics, vocabulary) is obtained from the matrix factorization.

To get the most relevant summary of a speech, it is necessary to know
which topics the speech is about. We use this intuition to map /
associate a top topics to a speech. We use the topic vector for every
speech and sort the vector in descending order to pick the top topic for
a speech. Every topic is associated with a vocabulary vector. The top
words associated with a topic can again be obtained by sorting the
topic-vocabulary vector in descending order.

Consider the example below. 

### Sentence Extraction for Summarization


