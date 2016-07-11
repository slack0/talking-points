
# Talking Points

An accurate summarization tool for politcal speeches and interviews.

## Introduction

Speech summarization is a necessity in many domains. It is needed in
cases where the audience needs to get a gist of a speech or an article
in a way that preserves the intent and the meaning of the speech. This
problem belongs to a class of problems under [Automatic
summarization](https://en.wikipedia.org/wiki/Automatic_summarization).
The main idea of summarization is to find a representative subset of
the data, which contains the information of the entire set.
Document summarization, tries to automatically create a representative
summary or abstract of the entire document, by finding the most informative
sentences.

## Summary Generation from Political Speeches

Summary generation is difficult and is context dependent.
The objective and the metrics of summarization process themselves are difficult to define in some situations. Extracting information that is representative of an entire discussion/topic/speech is nuanced as the speaker may convey meaning implicitly while referring to a past or a future discussion.  In this work, we consider the problem of summary extraction from documents/speeches based on the knowledge of what topics the speech is about.  

The main intuition here is that the inference about the topic is valuable in evaluating which parts of a document or a speech are relevant to it.  Using this intuition, this tool provides a summary of the most important topics of speeches based on speaking style (word usage) of a specific person (i.e., from a speech corpus). Based on topic extraction and sentence/document vectorization, the tool extracts most relevant / important sentences from a speech by ranking topic similarity to the sentence similarity.

Results show that this technique outperforms similar summarization techniques that rely only on sentence similarity.


