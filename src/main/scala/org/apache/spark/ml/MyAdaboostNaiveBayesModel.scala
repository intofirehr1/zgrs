package org.apache.spark.ml

import com.chinalife.classification.AdaboostNaiveBayesParams
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.classification.{MyNaiveBayesModel, ProbabilisticClassificationModel}
import org.apache.spark.ml.linalg.{DenseVector, Matrix, SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

/**
  * Created by hadoop on 2017/8/3 0003.
  */
class MyAdaboostNaiveBayesModel  (override val uid: String, val iternums: Int,
                                  val modelWeights: Array[Double], val modelArray: Array[MyNaiveBayesModel])
  extends ProbabilisticClassificationModel[Vector, MyAdaboostNaiveBayesModel]
    with AdaboostNaiveBayesParams with MLWritable{

  override val numClasses = modelArray(0).pi.size

  private def multinomialCalculation(features: Vector): Vector = {
    val result: Vector = new DenseVector(new Array(numClasses))
    for (i <- 0 until iternums) {
      val prob: Vector = modelArray(i).theta.multiply(features)
      prob.foreachActive { (index, value) => {
        prob.toArray(index) = value + modelArray(i).pi(index)
      }
      }
      result.toArray(prob.argmax) = result(prob.argmax) + modelWeights(i)
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
      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF
  }

  override def predictRaw(features: Vector): Vector = {
    multinomialCalculation1(features)
    //  TODO 返回类别+概率
  }
  def predictRaw1(features: Vector): Vector = {
    multinomialCalculation1(features)
  }

  /**
    *
    * @param features
    * @return
    */
  private def getProbValue(features: Vector): Vector = {
//    var result: Array[Double] = new Array[Double](numClasses)
    val result: Vector = new DenseVector(new Array(numClasses))
    var tmp : Array[Array[Double]] = new Array[Array[Double]](iternums)
    for (i <- 0 until iternums) {
      val prob: Vector = modelArray(i).theta.multiply(features)
      prob.foreachActive { (index, value) => {
        prob.toArray(index) = value + modelArray(i).pi(index)
      }
      }
      tmp(i) = raw2probabilityInPlace(prob).toArray.map( x =>
        x * modelWeights(i)
      )
    }
    for(i <- 0 until numClasses){
      val sums = tmp.map{x =>
        x.array.apply(i)
      }.sum

      result.toArray(i) = sums /1.0
    }

    result
  }

  private def multinomialCalculation1(features: Vector): Vector = {
    val result: Vector = new DenseVector(new Array(numClasses))
    for (i <- 0 until iternums) {
      val prob: Vector = modelArray(i).theta.multiply(features)
      prob.foreachActive { (index, value) => {
        prob.toArray(index) = value + modelArray(i).pi(index)
      }
      }
//      println("================= prop : " + prob.toArray.mkString("",",",""))
      result.toArray(prob.argmax) = result(prob.argmax) + modelWeights(i)
    }
//    println(" ==============  results " + result.toArray.mkString("",",",""))
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
        throw new RuntimeException("Unexpected error in AdaboostNaiveBayesModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)
  override def write: MLWriter = new MyAdaboostNaiveBayesModel.MyAdaboostNaiveBayesModelWriter(this)

}

object MyAdaboostNaiveBayesModel extends MLReadable[MyAdaboostNaiveBayesModel]{

  override def read: MLReader[MyAdaboostNaiveBayesModel] = new MyAdaboostNaiveBayesModelReader

  override def load(path: String): MyAdaboostNaiveBayesModel = super.load(path)

  /** [[MLWriter]] instance for [[MyAdaboostNaiveBayesModel]] */
  private[MyAdaboostNaiveBayesModel] class MyAdaboostNaiveBayesModelWriter(instance: MyAdaboostNaiveBayesModel) extends MLWriter {

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

  private class MyAdaboostNaiveBayesModelReader extends MLReader[MyAdaboostNaiveBayesModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[MyAdaboostNaiveBayesModel].getName

    override def load(path: String): MyAdaboostNaiveBayesModel = {
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

//      val Row(uid:String, iternums:Int, modelWeights: Array[Double], modelArray: Array[String]) =
//       data.select("uid","iternums","modelWeights","modelArray").head()

      val modelRealArray : Array[MyNaiveBayesModel] = new Array[MyNaiveBayesModel](keyedPairs._3.size)
      for(i <- 0 to keyedPairs._3.size-1){
        modelRealArray(i) = MyNaiveBayesModel.load(keyedPairs._4(i))
      }
      val model = new MyAdaboostNaiveBayesModel(keyedPairs._1, keyedPairs._2, keyedPairs._3, modelRealArray)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
