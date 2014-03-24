import cc.factorie.la.DenseTensor1
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.WeightsMapAccumulator
import cc.factorie.optimize.Example
import scala.collection.mutable.ArrayBuffer

class FastCBOWNegSamplingEmbeddingModel(override val opts : EmbeddingOpts) extends FastWordEmbeddingModel(opts) {
      val negative = opts.negative.value
      val window = opts.window.value
      val rng = new util.Random
      val sample = opts.sample.value.toDouble
      override def process(doc : String) : Int = {
            // given a document, below line splits by space and converts each word to Int (by vocab.getId) and filters out words not in vocab
            //var sen = new ArrayBuffer[Int]()
            val tokens = doc.split(' ')
            var sen = new Array[Int](tokens.size)
            var a = 0
            var b = 0
            var l = tokens.size
            var wordCount = 0
            var senLength = 0
            while (a < l) {
                val id = vocab.getId( tokens(a) )
                if(id != -1) {
                   wordCount += 1
                   if (sample == 0 || subSample(id) != -1) {
                     sen(senLength) = id
                     senLength += 1
                   }
                }
                a += 1
            }
           
            
              
            var senPosition = 0
            while (senPosition < senLength) {
                  val currWord = sen(senPosition)
                  // find the contexts
                  b = rng.nextInt(window)
                  val contexts =  new collection.mutable.ArrayBuffer[Int]
                  a = b
                  while (a < window * 2 + 1 - b) {
                       if (a != window ) {
                          val c = senPosition - window + a
                          if (c >=0 && c < senLength) 
                            contexts += sen(c)
                    }
                    a += 1
                  }
                       // make the examples
                     trainer.processExample(new FastCBOWNegSamplingExample(this, currWord, contexts, 1))
                     trainer.processExample(new FastCBOWNegSamplingExample(this, currWord, List(vocab.getRandWordId), -1))
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
class FastCBOWNegSamplingExample(model : FastCBOWNegSamplingEmbeddingModel, word : Int, contexts : Seq[Int], label : Int) extends Example{
  
  // to understand the gradient and objective refer to : http://arxiv.org/pdf/1310.4546.pdf
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
   
     val wordEmbedding = model.weights(word).value
     val contextEmbedding = new DenseTensor1(model.D, 0)
     var i = 0
     while (i < contexts.size) {
       contextEmbedding.+=( model.weights(contexts(i)).value )
       i += 1
     }
     //contexts.foreach(context => contextEmbedding.+=( model.weights(context).value ) ) 
    
     val score : Double = wordEmbedding.dot(contextEmbedding)
     val exp : Double = math.exp(-score) // TODO : pre-compute , costly operation
     
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
         var i = 0
         while (i < contexts.size) {
           gradient.accumulate(model.weights(contexts(i)), wordEmbedding, factor)
           i += 1
         }
        // contexts.foreach(context => gradient.accumulate(model.weights(context), wordEmbedding, factor) )
         gradient.accumulate(model.weights(word), contextEmbedding, factor)
    }
      
    
  }
}
