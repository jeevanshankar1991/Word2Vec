//package cc.factorie.app.nlp.embeddings;
import cc.factorie.optimize.Example
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.segment.DeterministicTokenizer
import cc.factorie.la.WeightsMapAccumulator
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.DenseTensor1

/*class MSSkipGramEmbeddingModel(override val opts : EmbeddingOpts) extends MSWordEmbeddingModel(opts) {
      val negative = opts.negative.value
      val window = opts.window.value
      val rng = new util.Random
      
      override def getExamplesFromSingleDocument(doc : String) : Seq[Example] = {
            val examples = new collection.mutable.ArrayBuffer[Example]
            // given a document, below line splits by space and converts each word to Int (by vocab.getId) and filters out words not in vocab
            val sen = doc.stripLineEnd.split(' ').map(word => vocab.getId(word)).filter(id => id != -1)
            val senLength = sen.size
            for (senPosition <- 0 until senLength) {
                  val currWord = sen(senPosition)
                  // find the contexts
                  val b = rng.nextInt(window) 
                  for (a <- b until window * 2 + 1 - b) if (a != window ) {
                       val c = senPosition - window + a
                       val contexts = new collection.mutable.ArrayBuffer[Int]
                       if (c >=0 && c < senLength) {
                           contexts += sen(c)
                       }
                       // make the examples
                       for (posContext <- contexts) {
                           examples += new MSSkipGramExample(this, contexts, currWord, posContext, 1)
                           (0 until negative).foreach( {
                               val negContext = vocab.getRandWordId
                               examples += new MSSkipGramExample(this, null, currWord, negContext, -1)
                           })
                       }
                  }
            }
            examples     
      }
}
class MSSkipGramExample(model : MSSkipGramEmbeddingModel, contexts : Seq[Int] = null,word : Int, context : Int, label : Int ) extends Example {
   def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
     val S = model.S
     val D = model.D
     val contextsEmbedding = new DenseTensor1(D, 0)
     contexts.foreach( context => contextsEmbedding.+=(model.weights(context)(0).value) )
     val sense = (0 until S+1).map(s => TensorUtils.cosineDistance(model.weights(s).value , y)).max._2
     val wordEmbedding = model.weights(word)(sense).value // access the word's embedding 
     val contextEmbedding = model.weights(context)(0).value // acess the context's embedding 
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
         gradient.accumulate(model.weights(word)(sense), contextEmbedding, factor)
         gradient.accumulate(model.weights(context)(0), wordEmbedding, factor)
    }
      
  }
}
}*/
