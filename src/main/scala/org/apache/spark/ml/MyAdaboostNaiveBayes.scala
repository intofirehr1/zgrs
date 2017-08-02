package org.apache.spark.ml

import com.chinalife.classification.AdaboostNaiveBayesParams
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._

/**
  * Created by hadoop on 2017/8/3 0003.
  */
class MyAdaboostNaiveBayes (override val uid: String)
  extends ProbabilisticClassifier[Vector, MyAdaboostNaiveBayes, MyAdaboostNaiveBayesModel]
    with AdaboostNaiveBayesParams {

  def this() = this(Identifiable.randomUID("MyAdaboostNaiveBayes"))

  setMaxIter(1)
  setThreshold(0.02)
  setSmoothing(1.0)
  setModelType("multinomial")
  setModelPath(Array[String](""))
//  setProb(Array[String](""))

  // method used by fit()
  override protected def train(dataset: Dataset[_]): MyAdaboostNaiveBayesModel = {

    val datasetSize = dataset.count().toInt
    val labelSize = dataset.select(${labelCol}).distinct().count()
    // setWeightCol("obsWeights")

    // 各子模型及其权重
    val modelWeights = new Array[Double]($(maxIter))
    val modelArray = new Array[MyNaiveBayesModel]($(maxIter))

    var alpha = 0.0

    // 初始化各样本等权重
    val dataWeight: (Double, Double, Double) => Double = (obsWeight: Double, labelIndex: Double, prediction: Double) => {
      if (labelIndex == prediction) {
        obsWeight
      }
      else {
        obsWeight * math.exp(alpha)
      }
    }
    val sqlfunc = functions.udf(dataWeight)
    val temp = dataset.withColumn("obsWeights", functions.lit(1.0))
    var i = 0
    var error1 = 2.0
    var error2 = 1.0   // && (error1 - error2) > $(threshold)
    var weightSum = datasetSize.toDouble * datasetSize

    val modelPaths = $(modelPath)
    // 可以设置为按照月份模型进行迭代
    val itersize = modelPaths.size
    val iter = 12

    val weightsArray : Array[Double] = Array[Double](0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.1,0.3,0.06)
    while (i < $(maxIter)) {

      // 加载月度模型进行训练
      var modelp = modelPaths(0)
      if(itersize > 1){
        modelp = modelPaths(i)
      }
      /* 进行多次迭代的月度模型*/
//      val naiveBayes = MyAdaboostNaiveBayes4MonthModel.load(modelp)

      /* 只进行一次naivebayes 的月度模型 */
      val naiveBayes = MyNaiveBayesModel.load(modelp)

      val temp1 = naiveBayes.transform(temp).cache()
//      print("=========== in MyAdaboostNaiveBayes")
//      temp1.show(100)
//      var error = temp1.select($(labelCol), $(predictionCol), "obsWeights").rdd.map(row => {
//        if (row(0) != row(1))
//          row.getDouble(2)
//        else
//          0.0
//      }
//      ).sum()/(datasetSize)
//      val t5 = System.nanoTime()
//      error1 = error2
//      error2 = error
//      println("============ threshold:"  + $(threshold))
//      if(error1-error < $(threshold)){
//         println("=========  error < 0.02")
//      }
//      println("================== error: " + error + ", labelSize: " + labelSize)
//      alpha = Math.log((labelSize - 1)*(1 - error) / error)
//
//      println("==============  alpha :" + alpha)
      modelWeights(i) = 1.0/$(maxIter)
      modelArray(i) = naiveBayes

//      // 更新权重
//      temp = temp.withColumn("obsWeights", sqlfunc(functions.col("obsWeights"), functions.col($(labelCol)), functions.col($(predictionCol))));
//      weightSum = temp.select("obsWeights").rdd.map(row => (row.getDouble(0))).sum()
//      println("===============  weightSum : " + weightSum + ", datasetSize: " + datasetSize)
//      temp = temp.drop($(predictionCol), $(rawPredictionCol), $(probabilityCol))
//      temp = temp.withColumn("obsWeights", functions.col("obsWeights")/(weightSum/datasetSize))
//      println("=============== tmp in i:"+ i + ",show data start ")
//      temp.show(10)
//      println("=============== alpha " + alpha)
//      println("=============== show data end ")
      i += 1
    }

    val adaboostNaiveBayesModel = new MyAdaboostNaiveBayesModel(uid, i, modelWeights, modelArray)
    for (elem <- adaboostNaiveBayesModel.modelWeights) {
      println("=================" + elem + ", model weight")
    }

    for (elem <- adaboostNaiveBayesModel.modelArray) {
//      println("=================" + elem.pi.toArray.mkString("",",","") + " : pi , " + elem.theta.toArray.mkString("",",","") + " : theta;")
    }
    println("===================="+ adaboostNaiveBayesModel.modelArray.size)

    adaboostNaiveBayesModel
  }

  // model parameters assignment
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setThreshold(value: Double): this.type = set(threshold, value)
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  def setModelType(value: String): this.type = set(modelType, value)
  def setModelPath(value: Array[String]): this.type = set(modelPath, value)
//  def setProb(value: Array[String]): this.type = set(prob, value)
 // def setWeightCol(value: String): this.type = set(weightCol, value)

  override def copy(extra: ParamMap): MyAdaboostNaiveBayes = defaultCopy(extra)

}

object MyAdaboostNaiveBayes{

}
