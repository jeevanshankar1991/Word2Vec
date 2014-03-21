import cc.factorie.model.WeightsSet
import cc.factorie.optimize.GradientOptimizer
import cc.factorie.util.FastLogging
import cc.factorie.optimize.Trainer
import cc.factorie.optimize.Example
import cc.factorie.la.SmartGradientAccumulator
import cc.factorie.util.LocalDoubleAccumulator
import cc.factorie.util.Threading

class FastHogwildTrainer(val weightsSet: WeightsSet, val optimizer: GradientOptimizer, val nThreads: Int = Runtime.getRuntime.availableProcessors(), val maxIterations: Int = 3)
  extends Trainer  {
  
  def processExample(e: Example): Unit = {
     val gradientAccumulator = new SmartGradientAccumulator
     val value = new LocalDoubleAccumulator()
     e.accumulateValueAndGradient(value, gradientAccumulator)
     optimizer.step(weightsSet, gradientAccumulator.getMap, value.value)
  }
  var iteration = 0
  def processExamples(examples: Iterable[Example]): Unit = {
    Threading.parForeach(examples.toSeq, nThreads)(processExample(_))
  }
  def isConverged = iteration >= maxIterations
}