package org.apache.spark.ml

import java.util

import com.chinalife.classification.AdaboostNaiveBayesParams
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.{Dataset, functions}


/**
  * Created by hadoop on 2017/8/3 0003.
  */
class MyAdaboostNaiveBayes4Month (override val uid: String)
  extends ProbabilisticClassifier[Vector, MyAdaboostNaiveBayes4Month, MyAdaboostNaiveBayes4MonthModel]
    with AdaboostNaiveBayesParams {

    def this() = this(Identifiable.randomUID("MyAdaboostNaiveBayes4Month"))

    setMaxIter(10)
    setThreshold(0.02)
    setSmoothing(1.0)
    setModelType("multinomial")
    setModelPath(Array[String](""))
  // setWeightCol("obsWeights")

    // method used by fit()
    override protected def train(dataset: Dataset[_]): MyAdaboostNaiveBayes4MonthModel = {

      val datasetSize = dataset.count().toInt
      val labelSize = dataset.select($ {
        labelCol
      }).distinct().count()

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
      var temp = dataset.withColumn("obsWeights", functions.lit(1.0))
      var i = 0
      var error1 = 2.0
      var error2 = 1.0 // && (error1 - error2) > $(threshold)
      var weightSum = datasetSize.toDouble * datasetSize

      val modelPaths = $(modelPath)
      // 可以设置为按照月份模型进行迭代
      val itersize = modelPaths.size
      val iter = 12

      while (i < $(maxIter)) {
        val naiveBayes = new MyNaiveBayes().setFeaturesCol($(featuresCol)).setLabelCol($(labelCol)).setWeightCol("obsWeights")
            .setPredictionCol($(predictionCol)).setModelType($(modelType)).setSmoothing($(smoothing)).fit(temp)
        naiveBayes.pi
        temp = naiveBayes.transform(temp).cache()
        print("=========== in MyAdaboostNaiveBayes4Month")
        temp.show(10)
        var error = temp.select($(labelCol), $(predictionCol), "obsWeights").rdd.map(row => {
          if (row(0) != row(1))
            row.getDouble(2)
          else
            0.0
        }
        ).sum() / (datasetSize)

        error1 = error2
        error2 = error
        println("============ threshold:" + $(threshold))
        if (error1 - error < $(threshold)) {
          println("=========  error < 0.02")
        }

        alpha = Math.log((labelSize - 1) * (1 - error) / error)
        println("================== error: " + error + ", labelSize: " + labelSize + ", alpha :" + alpha)
        modelWeights(i) = alpha
        modelArray(i) = naiveBayes

        // 更新权重
        temp = temp.withColumn("obsWeights", sqlfunc(functions.col("obsWeights"), functions.col($(labelCol)), functions.col($(predictionCol))));
        weightSum = temp.select("obsWeights").rdd.map(row => (row.getDouble(0))).sum()

        temp = temp.drop($(predictionCol), $(rawPredictionCol), $(probabilityCol))
        temp = temp.withColumn("obsWeights", functions.col("obsWeights")/(weightSum/datasetSize))

        println("===============  weightSum : " + weightSum + ", datasetSize: " + datasetSize + ", i : " + i + " , alpha : " + alpha)
        i += 1
      }

      val myAdaboostNaiveBayes4MonthModelModel = new MyAdaboostNaiveBayes4MonthModel(uid, i, modelWeights, modelArray)
      for (elem <- myAdaboostNaiveBayes4MonthModelModel.modelWeights) {
        println("=================" + elem + ", model weight")
      }

      for (elem <- myAdaboostNaiveBayes4MonthModelModel.modelArray) {
        println("=================" + elem.pi.toArray.mkString("", ",", "") + " : pi , " + elem.theta.toArray.mkString("", ",", "") + " : theta;")
      }
      println("====================" + myAdaboostNaiveBayes4MonthModelModel.modelArray.size)

      myAdaboostNaiveBayes4MonthModelModel
    }

    // model parameters assignment
    def setMaxIter(value: Int): this.type = set(maxIter, value)
    def setThreshold(value: Double): this.type = set(threshold, value)
    def setSmoothing(value: Double): this.type = set(smoothing, value)
    def setModelType(value: String): this.type = set(modelType, value)
    def setModelPath(value: Array[String]): this.type = set(modelPath, value)
    // def setWeightCol(value: String): this.type = set(weightCol, value)
    override def copy(extra: ParamMap): MyAdaboostNaiveBayes4Month = defaultCopy(extra)


  def getUnPerent(labelcnt: Array[(Double,Long)]):  java.util.HashMap[Double,Double] = {
    val labelWeight = new java.util.HashMap[Double,Double]
    val tmpWeight = new java.util.HashMap[Double,Double]
    var total = 0.0
    for (elem <- labelcnt) {
      total = total + elem._2
    }

    if(labelcnt.size==2){
      for(elem <- labelcnt){
        labelWeight.put(elem._1, 1-(elem._2/total))
      }
    }else {
      var lam = 10.0
      var tmpTotal = 0.0
      for(elem <- labelcnt){
        println(elem._1 + ", lable; " + total + ", total; " + ((elem._2/total))+", elem/total;"  + elem._2 + ", percent: " + ((1-(elem._2/total))/(elem._2/total)) + " , ======= count")
        val tp = ((1-(elem._2/total))/(elem._2/total))
        tmpWeight.put(elem._1, tp)
        tmpTotal = tmpTotal + tp
      }
      println( tmpTotal + ", ======= count tmpTotal")

      for (elem <- labelcnt) {
        println(elem._1 + ", lable; " + total + ", total; " + ((elem._2/total))+", elem/total;"  + elem._2 + ", percent: " +  tmpWeight.get(elem._1)/tmpTotal + " , ======= count")
        labelWeight.put(elem._1, tmpWeight.get(elem._1)/tmpTotal)
      }
    }
    labelWeight
  }

}


  object MyAdaboostNaiveBayes4Month {

}


class MyAdaboostNaiveBayes4MonthModel  (override val uid: String, val iternums: Int, val modelWeights: Array[Double], val modelArray: Array[MyNaiveBayesModel])
  extends ProbabilisticClassificationModel[Vector, MyAdaboostNaiveBayes4MonthModel]
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

  override def predictRaw(features: Vector): Vector = {
    multinomialCalculation1(features)
    //  TODO 返回类别+概率
  }
  def predictRaw1(features: Vector): Vector = {
    multinomialCalculation1(features)
  }

  private def multinomialCalculation1(features: Vector): Vector = {
    val result: Vector = new DenseVector(new Array(numClasses))
    for (i <- 0 until iternums) {
      val prob: Vector = modelArray(i).theta.multiply(features)
      prob.foreachActive { (index, value) => {
        prob.toArray(index) = value + modelArray(i).pi(index)
      }
      }
      println("================= prop : " + prob.toArray.mkString("",",",""))
      result.toArray(prob.argmax) = result(prob.argmax) + modelWeights(i)
    }
    println(" ==============  result " + result.toArray.mkString("",",",""))
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
  override def write: MLWriter = new MyAdaboostNaiveBayes4MonthModel.MyAdaboostNaiveBayes4MonthModelWriter(this)

}

object MyAdaboostNaiveBayes4MonthModel extends MLReadable[MyAdaboostNaiveBayes4MonthModel]{

  override def read: MLReader[MyAdaboostNaiveBayes4MonthModel] = new MyAdaboostNaiveBayes4MonthModelReader

  override def load(path: String): MyAdaboostNaiveBayes4MonthModel = super.load(path)

  /** [[MLWriter]] instance for [[MyAdaboostNaiveBayes4MonthModel]] */
  private[MyAdaboostNaiveBayes4MonthModel] class MyAdaboostNaiveBayes4MonthModelWriter(instance: MyAdaboostNaiveBayes4MonthModel) extends MLWriter {

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

  private class MyAdaboostNaiveBayes4MonthModelReader extends MLReader[MyAdaboostNaiveBayes4MonthModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[MyAdaboostNaiveBayes4MonthModel].getName

    override def load(path: String): MyAdaboostNaiveBayes4MonthModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      println("====================="+ data.columns.mkString("",",",""))
      data.show(10)
      // 取出数据，根据NaiveBayesModel 模型保存路径，在组装MyAdaboost 模型
      val keyedPairs = data.rdd.map{ x =>

        println("====================" + x.getSeq[Double](2).toArray)
        println("====================" + x.getList[String](3).toArray)
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
      val model = new MyAdaboostNaiveBayes4MonthModel(keyedPairs._1, keyedPairs._2, keyedPairs._3, modelRealArray)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
