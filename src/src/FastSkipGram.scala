//package cc.factorie.app.nlp.embeddings;
import cc.factorie.optimize.Example
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.segment.DeterministicTokenizer
import cc.factorie.la.WeightsMapAccumulator
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.DenseTensor1

class FastSkipGramNegSamplingEmbeddingModel(override val opts : EmbeddingOpts) extends FastWordEmbeddingModel(opts) {
       val negative = opts.negative.value
      val window = opts.window.value
      val rng = new util.Random
      val sample = opts.sample.value.toDouble
      override def process(doc : String) : Int = {
            // given a document, below line splits by space and converts each word to Int (by vocab.getId) and filters out words not in vocab
            var sen = doc.stripLineEnd.split(' ').map(word => vocab.getId(word)).filter(id => id != -1)
            val wordCount = sen.size
            
            // subsampling -> speed increase 
            if (sample > 0) 
              sen = sen.filter(id => subSample(id) != -1)
              
            val senLength = sen.size
            var senPosition = 0
            while (senPosition < senLength) {
                  val currWord = sen(senPosition)
                  // find the contexts
                  val b = rng.nextInt(window) // 
                  val contexts =  new collection.mutable.ArrayBuffer[Int]
                  var a = b
                  while (a < window * 2 + 1 - b) {
                       if (a != window ) {
                          val c = senPosition - window + a
                          if (c >=0 && c < senLength) 
                            contexts += sen(c)
                    }
                    a += 1
                  }
                       // make the examples
                  var i = 0
                  var nContext = contexts.size
                  while (i < nContext) {
                    trainer.processExample(new FastSkipGramNegSamplingExample(this, currWord, contexts(i), 1))
                    trainer.processExample(new FastSkipGramNegSamplingExample(this, currWord, vocab.getRandWordId, -1))
                    i += 1
                  }
                       /*for (n <- 0 until negative) {
                         val negContext = vocab.getRandWordId
                         trainer.processExample(new FastCBOWNegSamplingExample(this, currWord, Array(negContext), -1))
                       }*/
                 senPosition += 1     
            }
            return wordCount
      }
      
      def subSample(word : Int) : Int = {
          val ran = vocab.getSubSampleProb(word) // pre-computed to avoid sqrt call every time. Improvement of 10 secs on 100MB data ~ 15 MINs on 10GB
          //val cnt = vocab.getCount(word)
          //val ran = (math.sqrt(cnt / (sample * train_words)) + 1) * (sample * train_words) / cnt
          val real_ran = rng.nextInt(0xFFFF)/0xFFFF.toDouble
          if (ran < real_ran) {  return -1 }
          else return word
      }
}
class FastSkipGramNegSamplingExample(model : FastSkipGramNegSamplingEmbeddingModel, word : Int, context : Int, label : Int ) extends Example{
  
  // to understand the gradient and objective refer to : http://arxiv.org/pdf/1310.4546.pdf
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
     val wordEmbedding = model.weights(word).value // access the word's embedding 
     val contextEmbedding = model.weights(context).value // acess the context's embedding 
     val score : Double = wordEmbedding.dot(contextEmbedding)
     val exp : Double = math.exp(-score) // TODO : pre-compute exp table
     var objective : Double = 0.0
     var factor : Double = 0.0
     if (label == 1) { 
          objective = -math.log1p(exp)
          factor = exp/(1 + exp)
     }
     if (label == -1) {
          objective = -score -math.log1p(exp)
          factor = -1/(1 + exp)
     }     
    if (value ne null) value.accumulate(objective)
    if (gradient ne null) {
         gradient.accumulate(model.weights(word), contextEmbedding, factor)
         gradient.accumulate(model.weights(context), wordEmbedding, factor)
    }
      
  }
}

/*object SkipGramNegSamplingEmbedding {
     def main(args : Array[String]) {
          val opts = new EmbeddingOpts
          val skipGramModel = new SkipGramNegSamplingEmbeddingModel(opts)
          skipGramModel.buildVocab()
          skipGramModel.learnEmbeddings
          Distance.load(opts.output.value)
          Distance.play()
          // TODO : skipGramModel.reportAccuracy() 
          // cmd line : SkipGramNegSamplingEmbedding --train text8 --output text8_vectors --thread=${nproc} --window=5 --save-vocab text8_vocab
     }
}*/
