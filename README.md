
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


