package com.chinalife.model

import java.text.DecimalFormat
import java.util
import java.util.Calendar

import com.chinalife.util.Commons
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{AdaBoostNaiveBayes4Year, Pipeline}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, ml}

/**
  * Created by hadoop on 2017/8/2 0002.
  */
class NbModelTrain4Year {

  def print(totalCnt: Double, testCnt:Double, modelClass: String,
            evaluator: MulticlassMetrics,nbMetricsCats:BinaryClassificationMetrics) : Unit = {
    val nbPrCats = nbMetricsCats.areaUnderPR()   // Area under PR: 74.0522%
    val nbRocCats = nbMetricsCats.areaUnderROC() // Area under ROC: 60.5138%
    val nbAccuracy = totalCnt / testCnt
    println("========== 4 year predict : " + "confusionMatrix:" + evaluator.confusionMatrix + ", " + evaluator.accuracy)
    // Weighted stats
    println("========== 4 year predict : " + s"Weighted precision: ${evaluator.weightedPrecision}" +
      ", " + s"Weighted recall: ${evaluator.weightedRecall}" +
      ", " + s"Weighted F1 score: ${evaluator.weightedFMeasure}" +
      ", " + s"Weighted F1 score: ${evaluator.weightedFMeasure}" +
      ", " + s"Weighted false positive rate: ${evaluator.weightedFalsePositiveRate}")

    println("========== 4 year predict : " + ", nbTotalCorrect:" + totalCnt +
      ", testNbData cnt : " + testCnt  + ", nbAccuracy: " + nbAccuracy )

    println("========== 4 year predict : " + modelClass +
      ", " + f"Accuracy: ${nbAccuracy * 100}%2.4f%%\n" +
      ", " + f"Area under PR: ${nbPrCats * 100}%2.4f%%\n" +
      ", " + f"Area under ROC: ${nbRocCats * 100}%2.4f%%")
  }
}

object NbModelTrain4Year {

  def main(args: Array[String]): Unit = {
    if (args.size < 5) {
      System.err.println("params: 1、master 2、modelsavepath 3 maxIter 4 month_modelPath 5 labelType")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("gs_model_train_year").setMaster(args(0))
    // 设置运行参数： cpu, mem
    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    import spark.implicits._

    val cols = spark.table(Commons.whole_table_customs_year_pho).columns.mkString("", ",", "").toUpperCase

    // 数据信息  过滤掉多余数据 TODO
    val datasDF = spark.sqlContext.sql("SELECT " + cols + " FROM " + Commons.whole_table_customs_year_pho +
      " AS TP RIGHT JOIN (SELECT MIN(MIOS_TRANS_CODE) AS TMP_MIOS_CODE, ACC_CUST_MONTH AS TMP_MONTH " +
      " FROM " + Commons.whole_table_customs_year_pho + " AS TM " +
      " GROUP BY TM.TRANS_STAT, TM.BANKACC_ID_TYPE,TM.BANKACC_ID_NO,TM.ACC_CUST_MONTH) AS TL " +
      "ON TP.MIOS_TRANS_CODE=TL.TMP_MIOS_CODE AND TP.ACC_CUST_MONTH=TL.TMP_MONTH")

    // 特征信息大类信息 year
    val featuresCols = spark.table(Commons.whole_table_features_msg_src).columns.mkString("", ",", "").toUpperCase
    val featuresDF = spark.sqlContext.sql("SELECT " + featuresCols +
      " FROM " + Commons.whole_table_features_msg_src + " WHERE FEATURE_FLAG=1 AND FEATURE_MODEL_FLAG<>1")

    val tmpFeaturesTypeDF = featuresDF.collect()

    //标签处理，标签的形式：100111  100113  100115  100119  100122  100125  100129, 与数据分隔相同
    val newDatas = datasDF.rdd.map { line =>
      val features = line.get(line.size - 1).toString.split("_").sorted
      (line.mkString("", "\t", "\t") ++ features.mkString("", "\t", ""))
    }.map(_.split("\t")).map(p => Row.fromSeq(p.toSeq))

    var featuresColss: String = ""
    for (elem <- 1 to tmpFeaturesTypeDF.size) {
      featuresColss = featuresColss + "FEATURE" + elem + ","
    }

    val schemaString = datasDF.columns.mkString("", ",", "")

    val schemaFeatures = StructType((schemaString + "," +
      (featuresColss.substring(0, featuresColss.size - 1))).split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))

    val dataDF = spark.createDataFrame(newDatas, schemaFeatures)

    // 缓存数据集
    dataDF.cache()

    /** ================= 特征处理  start ================ */
    val stringColumns = featuresColss.substring(0, featuresColss.size - 1).split(",")
    val index_transformers: Array[org.apache.spark.ml.PipelineStage] = stringColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_INDEX")
    )
    // Add the rest of your pipeline like VectorAssembler and algorithm
    val index_pipeline = new Pipeline().setStages(index_transformers)
    val index_model = index_pipeline.fit(dataDF)
    val df_indexed = index_model.transform(dataDF)
    //encoding columns
    val indexColumns = df_indexed.columns.filter(x => x contains "INDEX")
    val one_hot_encoders: Array[org.apache.spark.ml.PipelineStage] = indexColumns.map(
      cname => new OneHotEncoder()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_VEC").setDropLast(false)
    )

    val pipeline = new Pipeline().setStages(index_transformers ++ one_hot_encoders)
    val model = pipeline.fit(dataDF)

    // TRANS_CODE + TRANS_BAT_NO 确定唯一
    val featuresCol: Array[String] = (Array[String]("TRANS_CODE", "TRANS_BAT_NO")) ++
      (Array[String]("TRANS_STAT")) ++ (featuresColss.substring(0, featuresColss.size - 1).split(",")) ++ (featuresColss.
      substring(0, featuresColss.size - 1).split(",").map(x => x + "_INDEX_VEC").array)

    val labelType = args(4).toInt
    val nbModelDatasTmp = model.transform(dataDF).select(featuresCol.head, featuresCol.tail: _*).rdd.map { x =>
      val featuresVals = Commons.getEncodeData4Train(x,(featuresColss.substring(0, featuresColss.size - 1)).split(","))
      var label = if (x(2).toString().toDouble == 1) 1.0 else 0.0
      if(labelType > 0){
        label = if(null != x(2).toString() && !"".equals(x(2).toString()) ) x(2).toString().toDouble else 0.0
      }
      (x(0).toString(), x(1).toString(), label, ml.linalg.Vectors.dense(featuresVals._1),featuresVals._2)
    }

    // 模型训练数据集缓存
    nbModelDatasTmp.cache()

    val nbModelData = nbModelDatasTmp.map( x => ( x._1, x._2, x._3, x._4))
    nbModelData.persist()
    // 特征向量转换信息记录
    val featureVectorData = nbModelDatasTmp.flatMap{ x =>
      var rows = Array[Row]()
      for (elem <- x._5) {
        rows = rows ++ Array(Row.fromSeq(elem.split(",").toSeq))
      }
      rows
    }
    // 保存特征向量转换参数
    Commons.saveFeatureVector4Year(spark, featureVectorData)

    val maxIter = args(2).toInt

    val modelPath = args(3).split(";")

    val pipelineDataSet = nbModelData.toDF("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT","FEATURES")
    //val indexer = new StringIndexer().setInputCol("TRANS_STAT").setOutputCol("labelIndex").fit(pipelineDataSet)
    val Array(train, test) = pipelineDataSet.randomSplit(Array(0.8, 0.2))

    /*  月度模型只迭代一次**/
//    val naiveBayes = new MyAdaboostNaiveBayes().setFeaturesCol("FEATURES").
    // setLabelCol("TRANS_STAT").setPredictionCol("prediction").setModelType("multinomial").
    // setSmoothing(1.0).setMaxIter(maxIter).setModelPath(modelPath)

    /* 计算获取月模型对应权重 TODO  */
    val weightsArray : Array[Double] =  getMonthWeights(maxIter) //Array[Double](0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.1,0.3,0.06)

    /*  月度模型迭代多次**/
    val naiveBayes = new AdaBoostNaiveBayes4Year().setFeaturesCol("FEATURES").
      setLabelCol("TRANS_STAT").setPredictionCol("prediction").setModelType("multinomial").
      setSmoothing(1.0).setMaxIter(maxIter).setModelPath(modelPath).setCustomWeights(weightsArray)


    val myadamodel = naiveBayes.fit(train)

//    val labelcnt = nbModelData.map(x => (x._3,1)).reduceByKey(_+_).collect()
//
//    /*** 设置 类别 权重**/
//    val labelWeight = getUnPerent(labelcnt)

    val result_tmp = myadamodel.transform(test)
//    println("================ results schema : " + result_tmp.printSchema())
    val results = result_tmp.select("TRANS_CODE", "TRANS_BAT_NO", "TRANS_STAT", "prediction", "probability", "rawPrediction", "prob")

//    // 查看是否有labels 的 数据
//    results.foreach( row =>
//    println("====================== results recod : " + row.getString(0) + ", " +
//      row.getString(1) + ", " + row.getDouble(2) + ", " + row.getDouble(3) + ", " +
//      row.getAs[Vector](4) + ", " + row.getAs[Vector](5) + ", percent: " +
//      row.getAs[DenseVector](4).toArray(row.getDouble(3).toInt) + ", prob percent: " +
//      row.getAs[DenseVector](6).toArray(row.getDouble(3).toInt) + ", prob all : " +
//      row.getAs[DenseVector](6))
//    )

    // TODO 归一化处理结果比例
    val predictResult = results.rdd.map(row => (row.getString(0), row.getString(1), row.getDouble(2), row.getDouble(3),
      parsePercent(row.getAs[DenseVector](6).toArray(row.getDouble(3).toInt))))

//    val percent = ModelUtil.getPercent(predictPercent)

    // 保存结果数据
    val resultDF = datasDF.join(predictResult.toDF("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT1", "TRAIN_LABEL", "TRAIN_ACCURACY"),
      Seq("TRANS_CODE","TRANS_BAT_NO"), "left_outer")

    resultDF.createTempView("tmp_result_train")

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_customs_year_train +
      " SELECT " +
      "" + datasDF.columns.mkString("",",","") +
      " ,TRAIN_LABEL,TRAIN_ACCURACY " +
      " FROM tmp_result_train")


    //保存模型前先删除模型目录
    Commons.deleteIfExists(spark, args(1))
    myadamodel.write.overwrite().save(args(1))


    val nbTotalCorrect = predictResult.map(x => if(x._3 == x._4) 1.0 else 0.0).sum()
    val predictR = predictResult.map(x => (x._3, x._4))
    val nbMetricsCats = new BinaryClassificationMetrics(predictR)
    val evaluator = new MulticlassMetrics(predictR)

    // 打印输出
    new NbModelTrain4Year().print(nbTotalCorrect,test.count(),myadamodel.getClass.getSimpleName,evaluator,nbMetricsCats)

    nbModelData.unpersist()
    dataDF.unpersist()
    spark.stop()

  }

  def getUnPerent(labelcnt: Array[(Double,Int)]): util.HashMap[Double,Double]={
    val labelWeight = new util.HashMap[Double,Double]
    val tmpWeight = new util.HashMap[Double,Double]
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


  def parsePercent(percent : Double ) : String = {
    val df: DecimalFormat = new DecimalFormat("######0.00")
    df.format(percent*100) + "%"
  }


//  def getPercentAll(denseVector: DenseVector) = {
//    val df: DecimalFormat = new DecimalFormat("######0.00")
//    val arrayList : util.ArrayList[String] = new util.ArrayList[String](denseVector.size)
//    val argmax_val = denseVector(denseVector.argmax)
//    var total = 1.0
//    denseVector.values.foreach ( x =>
//      if (x != argmax_val) total = total + math.exp(x - argmax_val)
//    )
//
//    for (elem <- 0 to denseVector.values.size-1) {
//      if(elem == argmax_val){
//        arrayList.add( df.format((1.0 / total)*100) + "%")
//      }else {
//        arrayList.add( df.format(((math.exp(denseVector(elem) - argmax_val)) / total) * 100) + "%")
//      }
//    }
//    arrayList
//  }

  /**
    * 计算获取月度模型权重，需要根据实际数据情况及经验参数进行设置
    * 目前只设置1 所有月都平均， 2 当前月比重稍大，其他月比重相同
    */
  def getMonthWeights(maxIter : Int) : Array[Double] ={
    val monthWeigths : Array[Double] = new Array[Double](maxIter)
    val calendar = Calendar.getInstance()
    val currentMonth = calendar.get(Calendar.MONTH) + 1
    println("=============== currentMonth : " + currentMonth)
    var x = 2
    if(maxIter == 12){
      if(currentMonth == maxIter){
        x = 1
      }
      for(i <- 0 until 12){
        if(i == currentMonth -1 ){
          monthWeigths(i) = 3.0 / 12
        } else if(i == currentMonth -1 -1){
          monthWeigths(i) = 2.0 / 12
        } else if(i == currentMonth -1 +1){
          monthWeigths(i) = 2.0 / 12
        }else {
          monthWeigths(i) = (1- (3.0/12) - x * (2.0 / 12)) / 9
        }
      }
    } else {
      for(i <- 0 until maxIter){
        monthWeigths(i) = 1.0/maxIter
      }
    }
    println("============================ monthWeigths : " + monthWeigths.mkString("",",","") )
    monthWeigths
  }
}

