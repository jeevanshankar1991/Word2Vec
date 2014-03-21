import cc.factorie.la.DenseTensor2
import java.io.RandomAccessFile
import cc.factorie.util.Threading

object Test {
   var  threads = 12
   var corpus = ""
   var train_words : Long = 0
   
   def main(args : Array[String]) {
     val opts = new EmbeddingOpts
     opts.parse(args)
     corpus = opts.corpus.value
     buildVocab()
     val files = (0 until threads).map(i => i)
     Threading.parForeach(files, threads)(processBigData(_))
     println("DONE")
   }
   
   def processBigData(id : Int) : Unit = {
          val randomAccessFile = new RandomAccessFile(corpus, "r")
          val fileLen = randomAccessFile.length()
          println(fileLen)
          randomAccessFile.seek(fileLen / threads * id)
          val lineItr = new LineReader(randomAccessFile)
          var word_count : Long = 0
          var end = true
          var nlines = 0
          while (lineItr.hasNext && end) {
              word_count += lineItr.next.split(' ').size  //process(lineItr.next)
              nlines += 1
              if (word_count > train_words / threads ) end = false
          }
          randomAccessFile.close()
          println(nlines)
      }
   def buildVocab(minFreq : Int = 5) : Unit = {
            val vocab = new VocabBuilder
            println("Building Vocab")
            val rd = new LineReader(new RandomAccessFile(corpus, "r"))
            
            while (rd.hasNext) {
               /*  SUPER SLOW . WHY ?
                *  val doc = new Document(line)
                 new DeterministicTokenizer.process(doc)
                 doc.tokens.foreach(token => vocab.addWordToVocab(token.string))
                 * 
                 */
                 val line = rd.next
                 line.stripLineEnd.split(' ').foreach(word => vocab.addWordToVocab(word))
            }
            vocab.sortVocab(minFreq) // removes words whose count is less than minCount and sorts by frequency
            vocab.buildSamplingTable // for getting random word from vocab in O(1) otherwise would O(log |V|)
            val V = vocab.size
            println("Vocab Size :" + V)
            train_words = vocab.trainWords()
            
      }
}