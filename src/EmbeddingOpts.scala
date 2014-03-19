//package cc.factorie.app.nlp.embeddings;
import cc.factorie.util.CmdOptions

class EmbeddingOpts extends CmdOptions{
  
   // Algorithm related
   val dimension = new CmdOption("size", 200, "INT", "size of word vectors")
   val window = new CmdOption("window", 5, "INT", "use <int> skip length between words")
   val threads = new CmdOption("threads", 12, "INT", "use <int> threads")
   val negative = new CmdOption("negative", 1, "INT", "use <int> number of negative examples")
   val minCount = new CmdOption("min-count", 5, "INT", "This will discard words that appear less than <int> times; default is 5")
   val cbow = new CmdOption("cbow", 0, "INT", "This will will run skip gram with negative sampling") // 1 would be SkipGram // default method is skipgram 
   
   // Optimization related (Don't change if you do not understand how vectors are initialized)
   val rate = new CmdOption("rate", 0.025, "DOUBLE", "learning rate for adaGrad")
   val delta = new CmdOption("delta", 0.1, "DOUBLE", "delta for adaGrad")
    
   // IO Related (MUST GIVE Options)
   val read_vocab_file = new CmdOption("read-vocab", "NONE", "STRING", "vocab file")
   val save_vocab_file = new CmdOption("save-vocab", "/home/jeevan/word2vec-read-only/text8_linebreak_vocabmine", "STRING", "save vocab file")
   val corpus = new CmdOption("train", "/home/jeevan/word2vec-read-only/text8_linebreak", "STRING", "train file")
   val output = new CmdOption("output", "/home/jeevan/word2vec-read-only/vectors_text8_linebreak.txt", "STRING", "Use <file> to save the resulting word vectors")
  
   // Vocabulary related
   // Maximum 20 * 0.7 = 14M words in the vocabulary
   val vocabHashSize = new CmdOption("vocab-hash-size", 20e6.toInt, "INT", "Vocabulary hash size" ) 
   val samplingTableSize = new CmdOption("sampling-table-size", 1e8.toInt, "INT", "Sampling Table size")
   
   
   // Debug (Print more information on terminal for debugging)
   val debug = new CmdOption("debug", 2 , "INT", "debug mode <int> : 2 for debug and 1 for without debug")
   
}
