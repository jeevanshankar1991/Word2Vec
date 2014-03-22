//package cc.factorie.app.nlp.embeddings;
import cc.factorie.model.Parameters
import cc.factorie.app.nlp.segment.DeterministicTokenizer
import cc.factorie.app.nlp.Document
import cc.factorie.optimize.Trainer
import cc.factorie.optimize.HogwildTrainer
import cc.factorie.optimize.AdaGrad
import cc.factorie.optimize.Example
import cc.factorie.model.Weights
import cc.factorie.model.WeightsMap
import cc.factorie.optimize.AdaGradRDA
import cc.factorie.la.DenseTensor1
import java.io.PrintWriter
import java.io.File
import scala.util.Random
import cc.factorie.optimize.HogwildTrainer
import java.io.RandomAccessFile
import cc.factorie.util.Threading
import java.io.FileReader

abstract class FastWordEmbeddingModel(val opts : EmbeddingOpts) extends Parameters {
    
      // Algo related
      val D = opts.dimension.value // default value is 200
      private val threads = opts.threads.value  //  default value is 12
      private val adaGradDelta = opts.delta.value // default value is 0.1
      private val adaGradRate = opts.rate.value //  default value is 0.025 
      private val minCount = opts.minCount.value
      private val maxCount = opts.maxCount.value 
      private val vocabHashSize = opts.vocabHashSize.value
      private val samplingTableSize = opts.samplingTableSize.value
      
      // IO Related
      private val corpus = opts.corpus.value
      private val outputFile = opts.output.value 
      
      // data structures
      var vocab : VocabBuilder = null
      var trainer : FastHogwildTrainer = null
      var optimizer : AdaGradRDA = null
      private var corpusLineItr : Iterator[String] = null
      var V : Int = 0
      var weights : Seq[Weights] = null
      var train_words : Long = 0 
     
      
      // debug info
      private var nLines  = 0 // aka nDoc
      private var totalLines = 0 // aka totalDocs 
      
      
      // Component-1
      def buildVocab(minFreq : Int = 5) : Unit = {
            vocab = new VocabBuilder(vocabHashSize, samplingTableSize, 0.7)
            println("Building Vocab")
            val tokenizer = new DeterministicTokenizer()
            for (line <- io.Source.fromFile(corpus).getLines) {
               /*  SUPER SLOW . WHY ?
                 val doc = new Document(line)
                 tokenizer.process(doc)
                 doc.tokens.foreach(token => vocab.addWordToVocab(token.string))
                 * 
                 */
                 line.stripLineEnd.split(' ').foreach(word => vocab.addWordToVocab(word))
                 println(totalLines)
                 totalLines += 1
            }
            vocab.sortVocab(minCount, maxCount) // removes words whose count is less than minCount and sorts by frequency
            vocab.buildSamplingTable // for getting random word from vocab in O(1) otherwise would O(log |V|)
            vocab.buildSubSamplingTable(opts.sample.value)
            V = vocab.size
            train_words = vocab.trainWords
            println("Vocab Size :" + V)
            if (opts.saveVocabFile.hasValue) {
              println("Saving Vocab")
              vocab.saveVocab(opts.saveVocabFile.value)
            }
            
      }
      // Component-2
      def learnEmbeddings() : Unit = {
          println("Learning Embeddings")
          optimizer = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate)    
          weights  =  (0 until V).map(i =>  Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0)))) // initialized using wordvec random
          optimizer.initializeWeights(this.parameters)
          //trainer = new HogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue,
          //                         logEveryN = -1, locksForLogging = false)
          trainer = new FastHogwildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)
           val files = (0 until threads).map(i => i)
           Threading.parForeach(files, threads)(processBigData(_))
          /*
           corpusLineItr = io.Source.fromFile(corpus).getLines
           var examples = getExamplesInBatch()
           while (examples.size > 0 && !trainer.isConverged) {
             println("# documents (lines) done : %d, Progress %f".format(nLines, nLines/totalLines.toDouble * 100) + "%")
             trainer.processExamples(examples)
             examples = getExamplesInBatch()
           } */
           println("DONE")
          //store()
      }
      // Component-3
      def store() {
         val out = new PrintWriter(new File(outputFile))
         out.println( "%d %d".format(V, D) )
         for (v <- 0 until V) {
              out.print(vocab.getWord(v))
              val embedding = weights(v).value
              for (d <- 0 until D)
                 out.print( " " + embedding(d))
              out.print("\n")
              out.flush()
         } 
         out.close()
      }
      
      // TODO : make this process parallel 
     /* def getExamplesInBatch(maxEg : Int = 1e5.toInt) : Seq[Example] = {
           var examples = new collection.mutable.ArrayBuffer[Example]
           while (examples.size < maxEg && corpusLineItr.hasNext) {
                  examples ++= getExamplesFromSingleDocument(corpusLineItr.next)
                  nLines += 1
           }
           examples
      }*/
      def processBigData(id : Int) : Unit = {
         /* val randomAccessFile = new RandomAccessFile(corpus, "r")
          val fileLen = randomAccessFile.length()
          println(fileLen)
          randomAccessFile.seek(fileLen / threads * id)
          val lineItr = new LineReader(randomAccessFile)
          *
          */
          val fileLen = new File(corpus).length
          val skipBytes : Long = fileLen / threads * id
          val lineItr = new FastLineReader(corpus, skipBytes)
          var word_count : Long = 0
          var end = true
          var nlines = 0
          val total_words_per_thread = train_words / threads
          while (lineItr.hasNext && end) {
              word_count +=  process(lineItr.next)
              nlines += 1
              if (id == 1 && nlines%5 == 0) {
               println("Progress : " + word_count / total_words_per_thread.toDouble * 100 + " %")
              }
              if (word_count > train_words / threads ) end = false
          }
          //randomAccessFile.close()
      }
      // override this function in your Embedding Model like SkipGramEmbedding or CBOWEmbedding
      def process(doc : String) : Int
      //def getExamplesFromSingleDocument(doc : String) : Seq[Example]
      
}
