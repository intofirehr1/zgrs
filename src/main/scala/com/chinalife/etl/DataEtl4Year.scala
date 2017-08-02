package com.iss.chinalife.sql

import java.text.SimpleDateFormat
import java.util
import java.util.Calendar

import com.chinalife.util.Commons
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{BisectingKMeans, BisectingKMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession, _}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks

class DataEtl4Year {

}

/**
  * Created by hadoop on 2017/8/1 0001.
  */
object DataEtl4Year {
  def main(args: Array[String]): Unit = {
    if (args.size < 6) {
      System.err.println("params: 1、master 2、year_id 3、month_id 4 k 5 runs ")
      System.exit(1)
    }

    // 设置运行参数： cpu, mem
    val conf = new SparkConf().setAppName("gs_data_etl_year").setMaster(args(0))

    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    val featureRangCols = spark.table(Commons.whole_table_feature_rang).columns.mkString("",",","").toUpperCase
    val featureRangDF = spark.sqlContext.sql("SELECT " + featureRangCols +
      " FROM " + Commons.whole_table_feature_rang)

    // 解析其他特征数据的参数(主要解析身份证，日期等)
    val etlDF = parseData(spark, featureRangDF)

    //  kmeans 特征计算，少做一张表，然后也同时做，count,
    val k = args(1).toInt
    val runs = args(2).toInt
    val kmpath = args(3)
    val flag = args(4).toInt
    val maxK = args(5).toInt
    val predictResult = generKM(spark, etlDF, k, maxK, runs, kmpath, flag)

    predictResult.createTempView("tmp_features")

    //    val cntDf = spark.sqlContext.sql("select count(1) from tmp_features")
    //    println(cntDf.rdd.first().get(0)+" , ===== predictResutl size ")
    // 在这里需要做一次group by count 计算用户的扣款次数，次数做为一个特征值处理，
    // 在做模型预测的时候，时间和次数其实是俩个参考值
    val cntFDF = spark.sqlContext.sql("select count(1) as KK_CNT,BANKACC_ID_TYPE as idType," +
      "BANKACC_ID_NO as idNo,TRANS_STAT as t_stat " +
      "from tmp_features  " +
      "group by BANKACC_ID_TYPE,BANKACC_ID_NO,TRANS_STAT")

    val resultDF = predictResult.join(cntFDF,
      predictResult("BANKACC_ID_TYPE") === cntFDF("idType") &&
        predictResult("BANKACC_ID_NO") === cntFDF("idNo") &&
        predictResult("TRANS_STAT") === cntFDF("t_stat"),
      "left_outer")

    resultDF.createTempView("tmp_feat_km")

    val kmCols = spark.sqlContext.table(Commons.whole_table_customs_yesr).columns.mkString("", ",", "").toUpperCase
    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_customs_yesr +
      " SELECT " + kmCols + " FROM tmp_feat_km")

    spark.stop()
  }

  /**
    * 解析特征值
    * @param spark
    * @return
    */
  def parseData(spark: SparkSession, featureRangDF: DataFrame): DataFrame = {
    val cols = spark.table(Commons.whole_table_sources_year_tmp).columns.mkString("", ",", "").toUpperCase()
    val tmpDF = spark.sqlContext.sql("SELECT " + cols + " FROM " + Commons.whole_table_sources_year_tmp)
    val year = Calendar.getInstance().get(Calendar.YEAR)
    val featuresRangArray = featureRangDF.collect()
    val etlRDD = tmpDF.rdd.map { row =>
      val accIdType = row.getAs[String]("BANKACC_ID_TYPE")
      val accIdNo = row.getAs[String]("BANKACC_ID_NO")
      val transAmnt = row.getAs[String]("TRANS_AMNT")
      val pldate = row.getAs[String]("PLNMIO_DATE")
      val mioDate = row.getAs[String]("MIO_DATE")
      val value = parseSexAndHome(accIdType, accIdNo, year)
      //      val age = value._3
      //      var home = value._1
      //      var sex = value._2
      val amntRang = parseAmntRang(transAmnt,featuresRangArray)
      val ageRang = parseAgeRang(value._3, featuresRangArray)
      val pldateMonth = parseDate(pldate)
      val dayRang = parseDayRang(pldateMonth._2.toString, featuresRangArray)
      // 返回etlDf的row
      (Row.fromSeq(row.toSeq ++ Seq(value._2, value._1, value._3, ageRang, "0",
        amntRang, "1", pldateMonth._1.toString, pldateMonth._2.toString,
        dayRang, "0")))
    }
    val featCols = "MIOS_TRANS_CODE,SYS_NO,TRANS_CODE,TRANS_BAT_NO,"+
      "TRANS_STAT,TRANS_BAT_SEQ,PLNMIO_REC_ID,BRANCH_BANK_ACC_NO,BANK_CODE,"+
      "BANK_SUB_CODE,BANK_SUB_NAME,BANK_PROV_CODE,BANK_CITY_CODE,BANK_ACC_TYPE,"+
      "BANK_ACC_NO,ACC_CUST_NAME,BANKACC_ID_TYPE,BANKACC_ID_NO,MGR_BRANCH_NO,"+
      "CNTR_NO,IPSN_NO,TRANS_CLASS,MIO_CLASS,MIO_ITEM_CODE,PLNMIO_DATE,MIO_DATE,"+
      "TRANS_AMNT,CUST_NO,GENERATE_DATE,UNITE_TRANS_CODE,EXT01,EXT02,EXT03,EXT04,"+
      "EXT05,BANK_TRANS_STAT,BANK_TRANS_DESC,MIO_CUST_NAME,GCLK_BRANCH_NO,"+
      "GCLK_CLERK_CODE,MIO_TX_NO,ACC_CUST_SEX,ACC_CUST_HOME,ACC_CUST_AGE,"+
      "ACC_CUST_AGERANG,ACC_CUST_INSTANCETYPE,ACC_CUST_AMNTRANG,ACOUNT_TIME_RANG,"+
      "ACC_CUST_MONTH,ACC_CUST_DAY,ACC_CUST_DAYRANG"

    //    val featCols = spark.table(Commons.table_customFeatKm).columns.mkString("", ",", "")

    val schemaFeatures = StructType(featCols.toUpperCase.split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))
    val etlDF = spark.createDataFrame(etlRDD, schemaFeatures)
    etlDF
  }


  /**
    * 从信息字段解析年龄和姓名及地区等信息
    * @param accIdType
    * @param accIdNo
    * @param year
    * @return
    */
  def parseSexAndHome(accIdType: String, accIdNo: String, year: Int) = {
    var home = "111111"
    var sex = "1"
    var age = "1"
    if (null != accIdType && !"".equals(accIdType) && !"null".equals(accIdType)) {
      if ("I".equals(accIdType)) {
        if (accIdNo.size == 15) {
          home = accIdNo.substring(0, 6)
          sex = ((accIdNo.substring(accIdNo.size - 2, accIdNo.size - 1).toInt) % 2).toString
          age = (year - ("19".concat(accIdNo.substring(6, 8))).toInt).toString
        } else if (accIdNo.size == 18) {
          home = accIdNo.substring(0, 6)
          sex = ((accIdNo.substring(accIdNo.size - 2, accIdNo.size - 1).toInt) % 2).toString
          age = (year - (accIdNo.substring(6, 10)).toInt).toString
        } else {
          // 身份证号不对的情况，数据直接删除
        }
      } else {
        // TODO 其他类型证件处理
      }
    } else {
      // TODO 证件类型为null 的情况，数据有误！
    }
    (home, sex, age)
  }

  /**
    * @param spark
    * @param msgDF
    * @param k
    * @param runs
    * @return
    */
  def generKM(spark: SparkSession, msgDF: DataFrame, k: Int, maxK:Int, runs: Int,
              kmpath: String, flag: Int): DataFrame = {
    //    val featuresKm = "ACC_CUST_AMNTRANG,ACC_CUST_SEX,ACC_CUST_HOME," +
    //      "ACC_CUST_AGERANG,ACC_CUST_DAY,ACC_CUST_DAYRANG"

    val featuresKm = "ACC_CUST_AMNTRANG,ACC_CUST_SEX," +
      "ACC_CUST_AGERANG,ACC_CUST_DAYRANG"

    val nbModelData = msgDF.rdd.map { row =>
      (Vectors.dense(getEncodeData2(row, featuresKm.split(","))),
        row.getAs("TRANS_CODE"), row.getAs("TRANS_BAT_NO"))
    }
    // 模型训练数据集缓存
    // nbModelData.cache()

    val trainRDD = nbModelData.map(x => x._1)

    trainRDD.cache()
    var kMeansModel = if (flag == 1) {
      new BisectingKMeans().setK(k).run(trainRDD)
      //        KMeans.train(trainRDD, i, runs+i)
    } else {
      BisectingKMeansModel.load(spark.sparkContext, kmpath)
      //        (KMeansModel.load(spark.sparkContext, kmpath))
    }

    //val bkm = new BisectingKMeans().setK(20)  // 6
    //val model1 = bkm.run(nbModelData)
    // Show the compute cost and the cluster centers  输出本次聚类操作的收敛性，此值越低越好
    println(s"Compute Cost: ${kMeansModel.computeCost(trainRDD)}")
    kMeansModel.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      println(s"Cluster Center ${idx}: ${center}")
    }

    val tmpschema = "TRANS_CODE_K,TRANS_BAT_NO_K,ACC_CUST_KM"
    val schemaFeatures = StructType(tmpschema.split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))

    // 输出每组数据及其所属的子集索引
    val kmRDD = nbModelData.map { vec =>
      val cluster = kMeansModel.predict(vec._1).toString
      println("this: [ " + vec._1.toArray.mkString("",",","") + "], is belong to "+ cluster)
      Row(vec._2, vec._3, kMeansModel.predict(vec._1).toString)
    }
    val kmDF = spark.createDataFrame(kmRDD, schemaFeatures)

    // feat 表和 聚类结果聚合，形成进一步的用户特征数据，然后才可以做用户画像 ZGRS_TMP.CUSTOM_FEAT_KM
    val predictResult = msgDF.join(kmDF, msgDF("TRANS_CODE") === kmDF("TRANS_CODE_K") && msgDF("TRANS_BAT_NO") === kmDF("TRANS_BAT_NO_K") , "left_outer")

    //    println(kmDF.rdd.count() + ", ===== count kmDf")
    //    println(msgDF.rdd.count() + ", ===== count msgDF")
    //    println(predictResult.rdd.count() + ", ===== count predictResult")

    if (flag == 1) {
      Commons.deleteIfExists(spark, kmpath)
      kMeansModel.save(spark.sparkContext, kmpath)
    }
    predictResult
  }

  def getEncodeData2(x: Row, featureCol: Array[String]): Array[Double] = {
    val tmpArray = ArrayBuffer[Double]()
    for (elem <- featureCol) {
      val tmpval = x.getAs[String](elem).toDouble
      //println("name: " + elem + ", value : " + tmpval)
      tmpArray += tmpval
    }
    tmpArray.toArray
  }


  /**
    * 解析日期
    * @param date
    * @return
    */
  def parseDate(date: String) = {
    // 分析拆分出数据中的月和日
    val ca = Calendar.getInstance()
    val sdf = new SimpleDateFormat("yyyy-MM-dd")
    val newDate = sdf.parse(date)
    ca.setTime(newDate)
    val month = ca.get(Calendar.MONTH) + 1
    val day = ca.get(Calendar.DATE)
    (month, day)
  }

  /**
    * 解析年龄
    * @param age
    * @return
    */
  def parseAgeRang(age: String, featuresRangArray: Array[Row]): String = {
    var tmpAge = 0 +""
    for (elem <- featuresRangArray) {
      val flag = elem.getAs[String]("FEATURE_RANG_FLAG")
      if(Commons.AGE_RANG.equals(flag)){
        val start = elem.getAs[String]("FEATURE_RANG_START").toDouble
        val end = elem.getAs[String]("FEATURE_RANG_END").toDouble
        if (null != age && !"".equals(age) && !"null".equals(age)) {
          if (age.toInt > start && age.toInt <= end) {
            tmpAge = elem.getAs[String]("FEATURE_RANG_CODE")
          }
        } else {
          tmpAge = 9999 + ""
        }
      }
    }
    tmpAge
  }

  /**
    * 解析金额
    * @param amnt
    * @return
    */
  def parseAmntRang(amnt: String, featuresRangArray: Array[Row]): String = {
    var tmpAmnt = 0+""
    for (elem <- featuresRangArray) {
      val flag = elem.getAs[String]("FEATURE_RANG_FLAG")
      if(Commons.AMNT_RANG.equals(flag)){
        val start = elem.getAs[String]("FEATURE_RANG_START").toDouble
        val end = elem.getAs[String]("FEATURE_RANG_END").toDouble
        if (null != amnt && !"".equals(amnt) && !"null".equals(amnt)) {
          if (amnt.toDouble > start && amnt.toDouble <= end) {
            tmpAmnt = elem.getAs[String]("FEATURE_RANG_CODE")
          }
        } else {
          tmpAmnt = 9999 + ""
        }
      }
    }
    tmpAmnt
  }

  /**
    * 解析日期范围
    * @param day
    * @return
    */
  def parseDayRang(day: String, featuresRangArray: Array[Row]): String = {
    var tmpDay = 0 +""
    for (elem <- featuresRangArray) {
      val flag = elem.getAs[String]("FEATURE_RANG_FLAG")
      if(Commons.DATE_RANG.equals(flag)){
        val start = elem.getAs[String]("FEATURE_RANG_START").toDouble
        val end = elem.getAs[String]("FEATURE_RANG_END").toDouble
        if (null != day && !"".equals(day) && !"null".equals(day)) {
          if (day.toInt > start && day.toInt <= end) {
            tmpDay = elem.getAs[String]("FEATURE_RANG_CODE")
          }
        } else {
          tmpDay = 9999 + ""
        }
      }
    }
    tmpDay
  }
}




