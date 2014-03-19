Word2Vec
========
Google's word2vec in Scala for better hacking and research. 


Tools for computing distributed representtion of words
------------------------------------------------------

Implementation of the Continuous Bag-of-Words (CBOW) and the Skip-gram model (SG), demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous
Bag-of-Words (CBOW) or the Skip-Gram model using negative sampling. Hierarchial Softmax support will be added soon. The user should to specify the following:
 - desired vector dimensionality . Default : 200
 - the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model. Default : 5
 - number of threads to use. Default : 12
 - output file to store the vectors. 
 - number of examples for negative sampling . Default : 1

Learning is done using Hogwild Trainer with ADAGrad Optimizer. The Delta is set to 0.1 and rate is set to 0.025. These hyper-parameters need not changed be for corpus. 

Format of Corpus
-------------------
Corpus is assumed to one big file (ranging from 100MB to 10GB). 
Each line in corpus file is assumed to be a document.   

Scripts
---------------------
1. The script demo-word.sh uses a small (100MB) text corpus from the web, and trains a small word vector model. After the training is finished, the user can interactively explore the similarity of the words.

2. The scipt demo-word-accuracy.sh uses a small (100MB) text corpus from the web, and trains a small word vector model. After the training is finished, the user can see the accuracy of the word vectors for various semantic and syntactic tasks.

Hacking
---------------------
WordEmbeddingModel is an abstract class providing abstractions like default implemenations of building vocab from corpus, saving vocab, learning embeddings via HogWildTrainer with Adagrad optimizer, making batch examples from corpus.

How to build own model - All you is to override one method called "getExamplesFromSingleDocument(doc : String) : Seq[Example]". Input to this method is a document (String) . See two default implemenations 
 - SkipGram Architecture - SkipGramNegSamplingEmbeddingModel
 - CBOW Architecture - CBOWNegSamplingEmbeddingModel

References 
------------------------------
To understand the gradient and objective functions , refer to 
- word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method Yoav Goldberg and Omer Levy http://arxiv.org/pdf/1402.3722v1.pdf
- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
