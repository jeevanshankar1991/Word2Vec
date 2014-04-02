Word2Vec
========
Google's word2vec in Scala for better hacking and research. 

Intro
------------------------------------------------------

Implementation of the Continuous Bag-of-Words (CBOW) and the Skip-gram model (SG), demo scripts.

Given a text corpus, Word2Vec learns a vector for every word in the vocab using the Continuous
Bag-of-Words (CBOW) or the Skip-Gram model using negative sampling. 
specify the following:
 - corpus to train on (train)
 - output file to store the vectors (output) . 
 - desired vector dimensionality (size). Default : 200
 - the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model (window). Default : 5
 - number of threads to use. Default (threads) : 12
 - number of examples for negative sampling (negative) . Default : 1
 - type of the model (cbow) : Default skipgram (cbow=0 => skipgram, cbow=1 => CBOW )
 - number of times the word should have occured atleast (min-count) . Default : 5
 - number of most frequent words to ignore (max-count). To re-iterate, it will ignore top frequent words in training. Default : 100 (Note: this option was not available on google's word2vec)
 - sub-sampling (sample) . Default : 1e-3
 - ada-grad rate (rate). Default : 0.025
 - ada-grade delta (delta). Default : 0.1 . (Note : google's word2vec uses the simple stochastic gradient descent. So, delta is not an option to them).  

Learning is done using Hogwild Trainer with ADAGrad Optimizer. The Delta is set to 0.1 and rate is set to 0.025. These hyper-parameters need not be changed be for corpus. 
Hierarchial Softmax support will be added soon. Generally, Negative Sampling gives better results and more scalable than Hierarchical SoftMax.

Format of Corpus
-------------------
Corpus is assumed to one big file (ranging from 100MB to 10GB). 
Each line in corpus file is assumed to be sentence.  
For eg, sample corpus 

word_11 word_12 word_13 ..... word_1n 

word_21 word_22 word_23 ..... word_2m
          
word_L1 word_L2 word_L3 ..... word_Lp 


Performance (as taken from https://code.google.com/p/word2vec/)
--------------
The training speed can be significantly improved by using parallel training on multiple-CPU machine (use the switch '-threads N'). The hyper-parameter choice is crucial for performance (both speed and accuracy), however varies for different applications. The main choices to make are:

- architecture: skip-gram (slower, better for infrequent words) vs CBOW (fast)
- the training algorithm: hierarchical softmax (better for infrequent words) vs negative sampling (better for frequent words, better with low dimensional vectors). Note : Right now, hierarchical soft-max is not provided but will added soon
- sub-sampling of frequent words: can improve both accuracy and speed for large data sets (useful values are in range 1e-3 to 1e-5). Note : I have also added an option (max-count), where you can ignore top frequent words in your training algorithm. 
- stop-words/most-frequent words : can improve the speed and accuracy improves on smaller data sets (~ 100MB) but couldn;t observe any significant increase in performance on very large data sets (~10GB )
- dimensionality of the word vectors: usually more is better, but not always
- context (window) size: for skip-gram usually around 10, for CBOW around 5


Scripts
---------------------
- ./demo-word.sh 
The script demo-word.sh uses a small (100MB) text corpus from the web, and trains a small word vector model. After the training is finished, the user can interactively explore the similarity of the words.

- ./demo-word-accuracy.sh 
The scipt demo-word-accuracy.sh uses a small (100MB) text corpus from the web, and trains a small word vector model. After the training is finished, the user can view the accuracy of the word vectors for various semantic and syntactic tasks.

Hacking
---------------------
1. WordEmbeddingModel is an abstract class providing abstractions like default implemenations of building vocab from corpus, saving vocab, learning embeddings via HogWildTrainer with Adagrad optimizer, making batch examples from corpus.

How to build own model - Override the method called "process(doc : String) : Int". Input to this method is a document (String) . Output of this method is number of words processes.  See two default implemenations to get more clarity.  
 - SkipGram Architecture - SkipGramEmbeddingModel
 - CBOW Architecture - CBOWEmbeddingModel

2. WordVec.scala is a client code with which you interact/talk to various models using various options provided by the user. See the code for more clarity . 

References 
------------------------------
To understand the gradient and objective functions , refer to 
- https://code.google.com/p/word2vec/
- word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method Yoav Goldberg and Omer Levy http://arxiv.org/pdf/1402.3722v1.pdf
- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
