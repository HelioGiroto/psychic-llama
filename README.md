psychic-llama 1.0 
=============

Visualization of NYC Innovation Space

This project aims to identify similar scientists. By _similar_ this project means scientist working in similar fields or with similar patterns of commercialization of their discoveries. 

We display the results of the analysis as an interactive scatter plot. 

To estimate similarity we use data from the:

 - [ ] professional web pages of the scientists and their laboratory
 - [ ] titles and abstracts of scientific publications indexed in PubMed or ISI Web of Knowledge
 - [ ] pattern of citations from ISI Web of Knowledge

Usage 
==============


Backend 
==============

  1. scrape_sinai.sh 
   	   1. Acquires the text (without markup) of faculty web pages. 
       1. Creates a folder for each university. Each file in the folder contains the text of the scientist's web page at the university. 
       1. Creates a text file _combined_ that contains all of the text in all of the other files in the directory after filtering out stopwords and removing as much HTML and Javascript as possible. 

  1. make->lda-topics.py 
  		1. Performs latent Dirichlet allocation on _combined_ and saves the top 10 to
  		_lda_topics_
  
  1. jaccard->lda.py 
  		1. Projects the documents from _combined_ onto orthogonalized versions of the basis vectors in _lda_topics_.  