object WordVec {
    def main(args : Array[String]) {
         val opts = new EmbeddingOpts
         opts.parse(args)
         val skipgram =  if (opts.cbow.value == 1) new FastCBOWNegSamplingEmbeddingModel(opts) else new FastSkipGramNegSamplingEmbeddingModel(opts)
         val st1 = System.currentTimeMillis()
         skipgram.buildVocab()
         val st = System.currentTimeMillis()
         println("time taken : " + (st-st1)/1000.0)
         skipgram.learnEmbeddings
         val en = System.currentTimeMillis() - st  
         //Distance.nearestNeighbours("/home/jeevan/word2vec-read-only/vectors_google.bin")
         println("time taken : " + en/1000.0)
         skipgram.store
         
    }
 }