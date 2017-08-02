package org.apache.spark.ml

import com.chinalife.classification.AdaboostNaiveBayesParams
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.classification.{MyNaiveBayesModel, ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

/**
  * Created by hadoop on 2017/8/5 0005.
  */
class AdaBoostNaiveBayes4Year (override val uid: String)
  extends ProbabilisticClassifier[Vector, AdaBoostNaiveBayes4Year, AdaBoostNaiveBayes4YearModel]
    with AdaboostNaiveBayesParams {

  def this() = this(Identifiable.randomUID("AdaBoostNaiveBayes4Year"))
  setMaxIter(1)
  setThreshold(0.02)
  setSmoothing(1.0)
  setModelType("multinomial")
  setModelPath(Array[String](""))
  setCustomWeights(Array[Double](0.0))

  // method used by fit()
  override protected def train(dataset: Dataset[_]): AdaBoostNaiveBayes4YearModel = {

    val datasetSize = dataset.count().toInt
    val labelSize = dataset.select(${labelCol}).distinct().count()

    // 各子模型及其权重
    val modelWeights = new Array[Double]($(maxIter))
    val modelArray = new Array[AdaBoostNaiveBayes4MonthModel]($(maxIter))

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
//    var error1 = 2.0
//    var error2 = 1.0   // && (error1 - error2) > $(threshold)
//    var weightSum = datasetSize.toDouble * datasetSize

    val modelPaths = $(modelPath)
    // 可以设置为按照月份模型进行迭代

    while (i < $(maxIter)) {

      // 加载月度模型进行训练
      var modelp = modelPaths(0)

      if($(maxIter) > 1){
        if(modelPaths.size == $(maxIter)){
          modelp = modelPaths(i)
        }
      }

      /* 进行多次迭代的月度模型*/
      // val naiveBayes = MyAdaboostNaiveBayes4MonthModel.load(modelp)

      /* 只进行一次naivebayes 的月度模型 */
      val naiveBayes = AdaBoostNaiveBayes4MonthModel.load(modelp)

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
//      println("=========== customWeights : " + $(customWeights).array.mkString("",",","") +
//        ",$(customWeights)(i):  " + $(customWeights)(i))
      modelWeights(i) =$(customWeights)(i)
      //modelWeights(i) = 1.0/$(maxIter)
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

    val yearModel = new AdaBoostNaiveBayes4YearModel(uid, i, modelWeights, modelArray)
    for (elem <- yearModel.modelWeights) {
      println("=================" + elem + ", model weight")
    }

//    for (elem <- yearModel.modelArray) {
//      //      println("=================" + elem.pi.toArray.mkString("",",","") + " : pi , " + elem.theta.toArray.mkString("",",","") + " : theta;")
//    }
    println("===================="+ yearModel.modelArray.size)

    yearModel
  }

  // model parameters assignment
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setThreshold(value: Double): this.type = set(threshold, value)
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  def setModelType(value: String): this.type = set(modelType, value)
  def setModelPath(value: Array[String]): this.type = set(modelPath, value)
  def setCustomWeights(value: Array[Double]): this.type = set(customWeights, value)
  override def copy(extra: ParamMap): AdaBoostNaiveBayes4Year = defaultCopy(extra)

}

object AdaBoostNaiveBayes4Year{}


/**
  *
  * @param uid
  * @param iternums
  * @param modelWeights
  * @param modelArray
  */
class AdaBoostNaiveBayes4YearModel (override val uid: String, val iternums: Int,
                               val modelWeights: Array[Double], val modelArray: Array[AdaBoostNaiveBayes4MonthModel])
  extends ProbabilisticClassificationModel[Vector, AdaBoostNaiveBayes4YearModel]
    with AdaboostNaiveBayesParams with MLWritable{

  override val numClasses = modelArray(0).modelArray(0).pi.size

  private def multinomialCalculation(features: Vector): Vector = {
    val result: Vector = new DenseVector(new Array(numClasses))
    for (i <- 0 until iternums) {
      val sub_result: Vector = new DenseVector(new Array(numClasses))
      val model = modelArray(i)
      for(j <- 0 until model.iternums){
        val sub_model = model.modelArray(j)
        val prob: Vector = sub_model.theta.multiply(features)
        prob.foreachActive { (index, value) => {
          prob.toArray(index) = value + sub_model.pi(index)
        }
        }
        sub_result.toArray(prob.argmax) = sub_result(prob.argmax) + model.modelWeights(j)
      }
      result.toArray(sub_result.argmax) = result(sub_result.argmax) + modelWeights(i)
    }
    result
  }


  /**
    * Transforms dataset by reading from [[featuresCol]], and appending new columns as specified by
    * parameters:
    *  - predicted labels as [[predictionCol]] of type `Double`
    *  - raw predictions (confidences) as [[rawPredictionCol]] of type `Vector`
    *  - probability of each class as [[probabilityCol]] of type `Vector`.
    *
    * @param dataset input dataset
    * @return transformed dataset
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".transform() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = dataset
    var numColsOutput = 0
    if ($(rawPredictionCol).nonEmpty) {
      val predictRawUDF = udf { (features: Any) =>
        predictRaw(features.asInstanceOf[Vector])
      }
      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF(col(getFeaturesCol)))
      numColsOutput += 1
    }
    if ($(probabilityCol).nonEmpty) {
      val probUDF = if ($(rawPredictionCol).nonEmpty) {
        udf(raw2probability _).apply(col($(rawPredictionCol)))
      } else {
        val probabilityUDF = udf { (features: Any) =>
          predictProbability(features.asInstanceOf[Vector])
        }
        probabilityUDF(col($(featuresCol)))
      }
      outputData = outputData.withColumn($(probabilityCol), probUDF)
      numColsOutput += 1
    }
    if ($(predictionCol).nonEmpty) {
      val predUDF = if ($(rawPredictionCol).nonEmpty) {
        udf(raw2prediction _).apply(col($(rawPredictionCol)))
      } else if ($(probabilityCol).nonEmpty) {
        udf(probability2prediction _).apply(col($(probabilityCol)))
      } else {
        val predictUDF = udf { (features: Any) =>
          predict(features.asInstanceOf[Vector])
        }
        predictUDF(col($(featuresCol)))
      }
      outputData = outputData.withColumn($(predictionCol), predUDF)
      numColsOutput += 1
    }

    if ($(prob).nonEmpty) {
      val probUDF = udf { (features: Any) =>
        getProbValue(features.asInstanceOf[Vector])
      }
      outputData = outputData.withColumn(getProb, probUDF(col(getFeaturesCol)))
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: AdaBoostNaiveBayes4YearModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF
  }

  override def predictRaw(features: Vector): Vector = {
//    multinomialCalculation(features)

    multinomialCalculation1(features)
  }
  def predictRaw1(features: Vector): Vector = {
    multinomialCalculation1(features)
  }

  def weightInPlace(weight : Array[Double]): Array[Double] = {
    val expw = weight.map(x => Math.exp(x))
    val sums = expw.sum
    expw.map(x => x / sums)
  }


  /**
    *
    * @param features
    * @return
    */
  private def getProbValue(features: Vector): Vector = {
    //    var result: Array[Double] = new Array[Double](numClasses)
    val result: Vector = new DenseVector(new Array(numClasses))
    val tmp : Array[Array[Double]] = new Array[Array[Double]](iternums)
    val weights = weightInPlace(modelWeights)
    for (i <- 0 until iternums) {
      val sub_result: Vector = new DenseVector(new Array(numClasses))
      val model = modelArray(i)
      val sub_tmp : Array[Array[Double]] = new Array[Array[Double]](model.iternums )
      // 对所有权重做归一化，然后再
      val sub_weights = weightInPlace(model.modelWeights)
      for(j <- 0 until model.iternums ) {
        val sub_model = model.modelArray(j)
        val prob: Vector = sub_model.theta.multiply(features)
        prob.foreachActive { (index, value) => {
          prob.toArray(index) = value + sub_model.pi(index)
        }
        }
        sub_tmp(j) = raw2probabilityInPlace(prob).toArray.map( x =>
          x * sub_weights(j)
        )

      }

//      println("&&&&&&&&&&&&& sub_tmp" + sub_tmp.mkString("",",",""))

      getSum(numClasses, sub_tmp, sub_result)

//      println("&&&&&&&&&&&&& sub_result" + sub_result.toArray.mkString("",",",""))
//      for(i <- 0 until numClasses){
//        val sums = sub_tmp.map{x =>
//          x.array.apply(i)
//        }.sum
//
//        sub_result.toArray(i) = sums /1.0
//      }


      tmp(i) = raw2probabilityInPlace(sub_result).toArray.map( x =>
        x * weights(i)
      )

    }
//    println("&&&&&&&&&&&&& tmp" + tmp.mkString("",",",""))

    getSum(numClasses, tmp, result)

//    println("&&&&&&&&&&&&& result" + result.toArray.mkString("",",",""))
//    for(i <- 0 until numClasses){
//      val sums = tmp.map{x =>
//        x.array.apply(i)
//      }.sum
//
//      result.toArray(i) = sums /1.0
//    }

//    println("&&&&&&&&&&&&& getprob result："+ result.toArray.mkString("",",",""))

    result
  }


  /**
    * 对相同类别的数据的预测结果求和， 之后在和所有的类别的预测结果做归一化
    * @param numClasses
    * @param src_array
    * @param result
    * @return
    */
  private def getSum(numClasses:Int, src_array:Array[Array[Double]], result:Vector) = {
    for(i <- 0 until numClasses){
      val sums = src_array.map{x =>
        x.array.apply(i)
      }.sum

      result.toArray(i) = sums /1.0
    }
    result
  }

  private def multinomialCalculation1(features: Vector): Vector = {
    val result: Vector = new DenseVector(new Array(numClasses))
    // 年度模型迭代次数，主要指迭代月
    for (i <- 0 until iternums) {
      val sub_result: Vector = new DenseVector(new Array(numClasses))
      val model = modelArray(i)
      val tmp : Array[Array[Double]] = new Array[Array[Double]](model.iternums)
      // 月度模型迭代，指adaboost + naive bayes
      for(j <- 0 until model.iternums){
        //
        val sub_model = model.modelArray(j)
        val prob: Vector = sub_model.theta.multiply(features)
        prob.foreachActive { (index, value) => {
          prob.toArray(index) = value + sub_model.pi(index)
        }
        }
        // println("================= prop : " + prob.toArray.mkString("",",",""))
        //1 只计算月度模型的子模型权重下的结果 prob [0.323234,0.676766]
        tmp(j) = prob.toArray.map { x =>
          x + model.modelWeights(j)
        }
//        println("============== tmp(j) " + j + " , " + tmp(j).array.mkString("",",",""))
      }

//      println("=========================  tmp array size : " + tmp.array.size)

//      tmp.array.foreach(x => println( "============================ x.array.size: " + x.array.size))
      // 2 计算 result result 为月度模型经过迭代后，计算的一个归一化的结果概率
      for(i <- 0 until numClasses){
        val sums = tmp.map{x =>
//          println("=======================  x.array : " + x.array.mkString("",",",""))
          x.array.apply(i)
        }.sum
        sub_result.toArray(i) = sums /1.0
      }
      //println(" ==============  sub model results " + sub_result.toArray.mkString("",",",""))
      // 计算权重及最终结果
      result.toArray(sub_result.argmax) = result(sub_result.argmax) + modelWeights(i)
    }
    //println(" ==============  model results " + result.toArray.mkString("",",",""))
    result
  }

  override def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        val maxLog = dv.values.max
        for (i <- 0 until size) {
          dv.values(i) = math.exp(dv.values(i) - maxLog)
        }
        val probSum = dv.values.sum

        for (i <- 0 until size) {
          dv.values(i) = dv.values(i) / probSum
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in AdaBoostNaiveBayes4YearModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)
  override def write: MLWriter = new AdaBoostNaiveBayes4YearModel.AdaBoostNaiveBayes4YearModelWriter(this)

}

object AdaBoostNaiveBayes4YearModel extends MLReadable[AdaBoostNaiveBayes4YearModel]{

  override def read: MLReader[AdaBoostNaiveBayes4YearModel] = new AdaBoostNaiveBayes4YearModelReader

  override def load(path: String): AdaBoostNaiveBayes4YearModel = super.load(path)

  /** [[MLWriter]] instance for [[MyAdaboostNaiveBayesModel]] */
  private[AdaBoostNaiveBayes4YearModel] class AdaBoostNaiveBayes4YearModelWriter(instance: AdaBoostNaiveBayes4YearModel) extends MLWriter {

    //    private case class Data(pi: Vector, theta: Matrix)
    private case class Data(uid: String, iternums: Int, modelWeights: Array[Double], modelArray: Array[String])
    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: pi, theta
      val modelPathArray = new Array[String](instance.modelWeights.size)
      val count = instance.modelWeights.size
      for(i <- 0 to count-1){
        instance.modelArray(i).write.overwrite().save(path+"/subNaiveBayesModel/"+i)
        modelPathArray(i) = path+"/subNaiveBayesModel/"+i
      }

      val data = Data(instance.uid, instance.iternums, instance.modelWeights, modelPathArray)
      val dataPath = new Path(path, "data").toString
      // 先保存所有模型，然后再将地址给个列表，地址指向模型
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class AdaBoostNaiveBayes4YearModelReader extends MLReader[AdaBoostNaiveBayes4YearModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[AdaBoostNaiveBayes4YearModel].getName

    override def load(path: String): AdaBoostNaiveBayes4YearModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      //      println("====================="+ data.columns.mkString("",",",""))
      //      data.show(10)
      // 取出数据，根据NaiveBayesModel 模型保存路径，在组装MyAdaboost 模型
      val keyedPairs = data.rdd.map{ x =>
        val uid = x.getAs[String]("uid")
        val iternums = x.getAs[Int]("iternums")
        val modelWeights = x.getSeq[Double](2).toArray
        val modelArray =  x.getSeq[String](3).toArray
        (uid,iternums,modelWeights,modelArray)
      }.first()

      val modelRealArray : Array[AdaBoostNaiveBayes4MonthModel] = new Array[AdaBoostNaiveBayes4MonthModel](keyedPairs._3.size)
      for(i <- 0 to keyedPairs._3.size-1){
        modelRealArray(i) = AdaBoostNaiveBayes4MonthModel.load(keyedPairs._4(i))
      }
      val model = new AdaBoostNaiveBayes4YearModel(keyedPairs._1, keyedPairs._2, keyedPairs._3, modelRealArray)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
