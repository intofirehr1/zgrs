package com.chinalife.etl

import java.util

import com.chinalife.util.Commons
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import scala.util.control.Breaks


class FeaturesExtraction4Year {
}

/**
  * Created by hadoop on 2017/8/1 0001.
  */
object FeaturesExtraction4Year {
  def main(args: Array[String]): Unit = {
    if (args.size < 1) {
      System.err.println("params: master modelflag")
      System.exit(1)
    }
    // 设置运行参数： cpu, mem
    val conf = new SparkConf().setAppName("gs_feature_extraction_year").setMaster(args(0))

    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    val cols = spark.table(Commons.whole_table_customs_yesr).columns.mkString("", ",", "").toUpperCase
    val dataDF = spark.sqlContext.sql("SELECT " + cols + " " +
      " FROM " + Commons.whole_table_customs_yesr)

    //val model_flag = args(1)
    var i: Int = 1
    val featuresMsgDF = spark.sqlContext.sql("SELECT feature_col_index, feature_col_name," +
      " feature_pcode, feature_desc FROM " + Commons.whole_table_features_msg_src + " WHERE FEATURE_FLAG=1 ")

    val featuresRangMsgCols = spark.table(Commons.whole_table_feature_rang).columns.mkString("", ",", "").toUpperCase
    val featuresRangMsgDF = spark.sqlContext.sql("SELECT " + featuresRangMsgCols +
      " FROM " + Commons.whole_table_feature_rang).collect()

    var dflist = new util.ArrayList[Row]()

    val featuresMsg = featuresMsgDF.rdd.collect()
    //val utils = new FeaturesExtractionUtils
    for (elem <- featuresMsg) {
      val featureNo = dataDF.rdd.map(r => parse(elem, r)).distinct.collect().zipWithIndex.toMap
      // 这块需要根据特征值，写入小类的范围或取值区间的说明
      val tmp1 = detailFeaturesCode(featureNo, elem.getAs(3), i, elem.getAs(2).toString.toInt, dflist, spark, featuresRangMsgDF)
      i = tmp1._2
      dflist = tmp1._1
    }
    println(dflist.size())

    val featureTable = "feature_id,feature_code,feature_value,feature_name,feature_pcode,feature_ptype,feature_desc,feature_weight";
    val schemaFeatures = StructType(featureTable.split(",").map(fieldName => StructField(fieldName, StringType, true)))
    val resultDF = spark.createDataFrame(dflist, schemaFeatures)

    resultDF.createTempView("tmp_features")

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_features_year + " SELECT " +
      "feature_id,feature_code,feature_value,feature_name," +
      "feature_pcode,feature_ptype,feature_desc,feature_weight " +
      "FROM tmp_features")

    spark.stop()

  }

  def parse(line: Row, r: Row):String = {
    val value =  r.getAs[String](line.getAs[String](1))
    var result = "0"
    if(null != value && !"".equals(value)&& !"null".eq(value)){
      result = value.toString
    }
    result
  }

  /**
    * 形成特征数据
    * @param featuresNo
    * @param featureName
    * @param i
    * @param pCode
    * @param dflist
    * @return
    */
  def detailFeaturesCode (featuresNo: scala.collection.immutable.Map[String, Int],
                          featureName: String, i : Int, pCode: Int,
                          dflist : util.ArrayList[Row],spark:SparkSession, rangeMsg: Array[Row]) = {
    var cnt = i
    for (elem <- featuresNo) {
      var flag = true
      val loop = new Breaks
      loop.breakable {
        for (featuresRang <- rangeMsg) {
          if (pCode == featuresRang.getAs[String]("FEATURE_CODE").toInt && (elem._1.equals(featuresRang.getAs[String]("FEATURE_RANG_CODE")))) {
            dflist.add(Row(cnt.toString, (pCode + elem._2 + 1).toString, elem._1,
              elem._1, pCode.toString, featureName, featuresRang.getAs[String]("FEATURE_RANG_DESC"), 1 + ""))
            cnt += 1
            flag = false
            loop.break
          }
        }
      }
      if(flag){
        dflist.add(Row(cnt.toString, (pCode + elem._2 + 1).toString, elem._1,
          elem._1, pCode.toString, featureName, elem._1, 1 + ""))
        cnt += 1
      }

    }
    (dflist, cnt)
  }

}


