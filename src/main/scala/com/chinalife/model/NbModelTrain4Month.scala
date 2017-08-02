package com.chinalife.model

import java.util

import com.chinalife.util.{Commons, Label2User}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{AdaBoostNaiveBayes4Month, Pipeline}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, ml}

/**
  * Created by hadoop on 2017/8/2 0002.
  */
class NbModelTrain4Month {

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


object NbModelTrain4Month{
  val label2UserList = new util.ArrayList[Row]()
  val mylabel = "TRANS_STAT"

  def main(args: Array[String]): Unit = {
    if (args.size < 4) {
      System.err.println("params: 1、master 2、modelsavepath 3 maxiter 4 labeltype")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("gs_model_train_month").setMaster(args(0))
    // 设置运行参数： cpu, mem
    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    import spark.implicits._

    val cols = spark.table(Commons.whole_table_customs_pho).columns.mkString("",",","").toUpperCase

    // 数据信息  过滤掉多余数据 TODO
    val datasDF =spark.sqlContext.sql("SELECT " + cols + " FROM " + Commons.whole_table_customs_pho +
      " WHERE MIOS_TRANS_CODE IN (SELECT min(MIOS_TRANS_CODE)" +
      " FROM " + Commons.whole_table_customs_pho + " GROUP BY TRANS_STAT,BANKACC_ID_TYPE,BANKACC_ID_NO)")

    // 特征信息大类信息
    val featuresCols = spark.table(Commons.whole_table_features_msg_src).columns.mkString("",",","").toUpperCase
    val featuresDF = spark.sqlContext.sql("SELECT " + featuresCols +
      " FROM " + Commons.whole_table_features_msg_src + " WHERE FEATURE_FLAG=1 AND FEATURE_MODEL_FLAG<>2")

    val tmpFeaturesTypeDF = featuresDF.collect()

    //标签处理，标签的形式：100111  100113  100115  100119  100122  100125  100129, 与数据分隔相同
    val newDatas = datasDF.rdd.map { line =>
      // 需要对特征的顺序做一个调整
      val features = line.get(line.size - 1).toString.split("_").sorted
      (line.mkString("", "\t", "\t") ++ features.mkString("", "\t", ""))
    }.map(_.split("\t")).map( p => Row.fromSeq(p.toSeq))

    var featuresColss : String = ""
    for (elem <- 1 to tmpFeaturesTypeDF.size) {
      featuresColss = featuresColss + "FEATURE" + elem + ","
    }

    val schemaString = datasDF.columns.mkString("",",","")

    val schemaFeatures = StructType((schemaString + "," +
      (featuresColss.substring(0,featuresColss.size-1))).split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))

    val dataDF = spark.createDataFrame(newDatas, schemaFeatures)

    // 缓存数据集
    dataDF.cache()

    /**  ================= 特征处理  start ================*/
    //indexing columns 特征列  Array[String]
    val stringColumns = featuresColss.substring(0, featuresColss.size-1).split(",")

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
    val indexColumns  = df_indexed.columns.filter(x => x contains "INDEX")
    val one_hot_encoders: Array[org.apache.spark.ml.PipelineStage] = indexColumns.map(
      cname => new OneHotEncoder()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_VEC").setDropLast(false)
    )

    val pipeline = new Pipeline().setStages(index_transformers ++ one_hot_encoders)
    val model = pipeline.fit(dataDF)

    // TRANS_CODE + TRANS_BAT_NO 确定唯一
    val featuresCol : Array[String] = (Array[String]("TRANS_CODE", "TRANS_BAT_NO")) ++
      (Array[String]("TRANS_STAT")) ++ (featuresColss.substring(0, featuresColss.size-1).split(",")) ++ (featuresColss.
      substring(0, featuresColss.size-1).split(",").map(x => x + "_INDEX_VEC").array)

    val labelType = args(3).toInt
    val nbModelDatas = model.transform(dataDF).select(featuresCol.head, featuresCol.tail: _*)
      .map { x =>
        val featuresVals =  Commons.getEncodeData4Train(x, (featuresColss.substring(0,featuresColss.size-1)).split(","))
        var label = if(x(2).toString().toDouble == 1) 1.0 else 0.0
        if(labelType > 0){
          label = if(null != x(2).toString() && !"".equals(x(2).toString()) ) x(2).toString().toDouble else 0.0
        }
        val features =  ml.linalg.Vectors.dense(featuresVals._1)
        (x(0).toString(), x(1).toString(), label, features, featuresVals._2)
      }

    // 模型训练数据集缓存
    nbModelDatas.cache()

    val nbModelData = nbModelDatas.map( x => ( x._1, x._2, x._3, x._4))
    /**  ================= 特征处理  end  ================*/

    // 特征向量转换信息记录
    val featureVectorData = nbModelDatas.rdd.flatMap{ x =>
      var rows = Array[Row]()
      for (elem <- x._5) {
        rows = rows ++ Array(Row.fromSeq(elem.split(",").toSeq))
      }
      rows
    }

    // 保存特征向量转换参数
    Commons.saveFeatureVector4Month(spark, featureVectorData)

    val Array(train, test) = nbModelData.toDF("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT","FEATURES").randomSplit(Array(0.8, 0.2))

    /** 设置 naivebayes 算法递进迭代，提高预测正确率 **/
    val maxIter = args(2).toInt
    val myNaiveBayes = new AdaBoostNaiveBayes4Month().setFeaturesCol("FEATURES").setLabelCol("TRANS_STAT")//.setWeightCol("obsWeights")
      .setPredictionCol("prediction").setModelType("multinomial").setSmoothing(1.0).setMaxIter(maxIter)
    val nbModel = myNaiveBayes.fit(train)


    /**  单独naivebayes 算法 执行，只执行一次 **/
    /*val myNaiveBayes = new MyNaiveBayes().setFeaturesCol("FEATURES").setLabelCol("TRANS_STAT")//.setWeightCol("obsWeights")
      .setPredictionCol("prediction").setModelType("multinomial").setSmoothing(1.0)
    val nbModel = myNaiveBayes.fit(train)*/


    val predictResult = nbModel.transform(test).select("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT","prediction", "probability").
      rdd.map(row =>
      ( row.getString(0), row.getString(1), row.getDouble(2), row.getDouble(3),
        row.getAs[Vector](4).toArray.apply(row.getDouble(3).toInt)
        )
    )
    predictResult.foreach( row =>
      println(row._1 + ":TRANS_CODE, "+ row._2 + ":TRANS_BAT_NO," + row._3 + ":TRANS_STAT, " +
        row._4 +":prediction,"+ row._5 +":probability,   ======= records ")
    )

    // 保存结果数据
    val resultDF = datasDF.join(predictResult.toDF("TRANS_CODE","TRANS_BAT_NO","TRANS_STAT1", "TRAIN_LABEL", "TRAIN_ACCURACY"),
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

    val nbTotalCorrect = predictResult.map(x => if(x._3 == x._4) 1 else 0).sum()
    val predictR = predictResult.map(x => (x._3, x._4))
    val nbMetricsCats = new BinaryClassificationMetrics(predictR)
    val evaluator = new MulticlassMetrics(predictR)

    // 打印输出
    new NbModelTrain4Month().print(nbTotalCorrect,test.count(),nbModel.getClass.getSimpleName,evaluator,nbMetricsCats)


    nbModelDatas.unpersist()
    dataDF.unpersist()
    spark.stop()
  }

  /**
    * 保存用户信息和预测结果
    * @param label2User
    */
  def saveData(label2User: Label2User): Unit ={
    label2UserList.add(Row(label2User.trans_code, label2User.trans_bat_no, label2User.label.toString))
  }

}
