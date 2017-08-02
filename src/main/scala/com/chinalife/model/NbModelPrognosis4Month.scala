package com.chinalife.model

import com.chinalife.util.Commons
import org.apache.spark.ml.AdaBoostNaiveBayes4MonthModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, ml}

/**
  * Created by hadoop on 2017/8/2 0002.
  */
class NbModelPrognosis4Month {

  def print(totalCnt: Double, testCnt:Double, modelClass: String,
            evaluator: MulticlassMetrics,nbMetricsCats:BinaryClassificationMetrics) : Unit = {
    val nbPrCats = nbMetricsCats.areaUnderPR()   // Area under PR: 74.0522%
    val nbRocCats = nbMetricsCats.areaUnderROC() // Area under ROC: 60.5138%
    val nbAccuracy = totalCnt / testCnt
    println("========== 4 month predict : " + "confusionMatrix:" + evaluator.confusionMatrix + ", " + evaluator.accuracy)
    // Weighted stats
    println("========== 4 month predict : " + s"Weighted precision: ${evaluator.weightedPrecision}" +
      ", " + s"Weighted recall: ${evaluator.weightedRecall}" +
      ", " + s"Weighted F1 score: ${evaluator.weightedFMeasure}" +
      ", " + s"Weighted F1 score: ${evaluator.weightedFMeasure}" +
      ", " + s"Weighted false positive rate: ${evaluator.weightedFalsePositiveRate}")

    println("========== 4 month predict : " + ", nbTotalCorrect:" + totalCnt +
      ", testNbData cnt : " + testCnt  + ", nbAccuracy: " + nbAccuracy )

    println("========== 4 month predict : " + modelClass +
      ", " + f"Accuracy: ${nbAccuracy * 100}%2.4f%%\n" +
      ", " + f"Area under PR: ${nbPrCats * 100}%2.4f%%\n" +
      ", " + f"Area under ROC: ${nbRocCats * 100}%2.4f%%")
  }

}

object NbModelPrognosis4Month {
  def main(args: Array[String]): Unit = {
    if (args.size < 3) {
      System.err.println("params: master modelpath")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("gs_model_predict_month").setMaster(args(0))
    // 设置运行参数： cpu, mem

    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    import spark.implicits._

    val customPhoCols = spark.table(Commons.whole_table_customs_pho).columns.mkString("",",","").toUpperCase

    val datasDF = spark.sqlContext.sql("SELECT " + customPhoCols + " FROM " + Commons.whole_table_customs_pho)

    // 特征信息大类信息
    val featuresCols = spark.table(Commons.whole_table_features_msg_src).columns.mkString("",",","").toUpperCase
    val featuresDF = spark.sqlContext.sql("SELECT " + featuresCols +
      " FROM " + Commons.whole_table_features_msg_src + " WHERE FEATURE_FLAG=1 AND FEATURE_MODEL_FLAG<>2")

    val tmpFeaturesTypeDF = featuresDF.collect()
    //标签处理，标签的形式：100111  100113  100115  100119  100122  100125  100129, 与数据分隔相同
    val newDatas = datasDF.rdd.map { line =>
      val features = line.get(line.size - 1).toString.split("_").sorted
      (line.mkString("", "\t", "\t") ++ features.mkString("", "\t", ""))
    }.map(_.split("\t")).map( p => Row.fromSeq(p.toSeq))

    var featuresColss : String = ""
    for (elem <- 1 to tmpFeaturesTypeDF.size) {
      featuresColss = featuresColss + "FEATURE" + elem + ","
    }

    val schemaString = datasDF.columns.mkString("",",","").toUpperCase
    val schemaFeatures = StructType((schemaString + "," +
      (featuresColss.substring(0,featuresColss.size-1))).split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))

    val dataDF = spark.createDataFrame(newDatas, schemaFeatures)

    // 缓存数据集
    dataDF.cache()

    val featureVectorCols = spark.table(Commons.whole_table_features_vectors).columns.mkString("",",","").toUpperCase
    val featVector = spark.sqlContext.sql("SELECT " + featureVectorCols +
      " FROM " + Commons.whole_table_features_vectors).collect()

    val labelType = args(2).toInt
    val nbModelData = dataDF.rdd.map{ x =>
      val trans_stat = x.getAs[String]("TRANS_STAT")
      var label = if(trans_stat.toDouble == 1) 1.0 else 0.0
      if(labelType > 0){
        label = if(null != trans_stat && !"".equals(trans_stat) )
          trans_stat.toDouble else 0.0
      }

      val features =  ml.linalg.Vectors.dense(Commons.getEncodeData4Model(x,
        (featuresColss.substring(0,featuresColss.size-1)).split(","), featVector))

      (x.getAs[String]("TRANS_CODE"), x.getAs[String]("TRANS_BAT_NO"),label, features )
    }

    // 模型训练数据集缓存
    nbModelData.cache()
    /** 年度模型可进行迭代的模型*/
    val nbModel = AdaBoostNaiveBayes4MonthModel.load(args(1))

    /** 年度模型 只执行一次 naiveBayes */
//    val nbModel = MyNaiveBayesModel.load(args(1))

    val prognosisResult = nbModel.transform(nbModelData.toDF("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT","FEATURES")).
      select("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT","prediction", "probability").
      rdd.map(row =>
      ( row.getString(0), row.getString(1), row.getDouble(2), row.getDouble(3),
        row.getAs[Vector](4).toArray.apply(row.getDouble(3).toInt)
        )
    )

    prognosisResult.foreach( row =>
      println(row._1 + ":TRANS_CODE, "+ row._2 + ":TRANS_BAT_NO," + row._3 + ":TRANS_STAT, " +
        row._4 +":prediction,"+ row._5 +":probability,   ======= records ")
    )

    // 保存结果数据
    val resultDF = datasDF.join(prognosisResult.toDF("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT1", "TRAIN_LABEL", "TRAIN_ACCURACY"),
      Seq("TRANS_CODE","TRANS_BAT_NO"), "left_outer")

    resultDF.createTempView("tmp_result_train")

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_customs_train +
      " SELECT " +
      "" + datasDF.columns.mkString("",",","") +
      " ,TRAIN_LABEL,TRAIN_ACCURACY " +
      " FROM tmp_result_train")


    //保存模型前先删除模型目录
    Commons.deleteIfExists(spark, args(1))
    nbModel.write.overwrite().save(args(1))

    val nbTotalCorrect = prognosisResult.map(x => if(x._3 == x._4) 1 else 0).sum()
    val predictR = prognosisResult.map(x => (x._3, x._4))
    val nbMetricsCats = new BinaryClassificationMetrics(predictR)
    val evaluator = new MulticlassMetrics(predictR)

    // 打印输出
    new NbModelTrain4Month().print(nbTotalCorrect,prognosisResult.count(),nbModel.getClass.getSimpleName,evaluator,nbMetricsCats)

    nbModelData.unpersist()
    dataDF.unpersist()
    spark.stop()
  }
}