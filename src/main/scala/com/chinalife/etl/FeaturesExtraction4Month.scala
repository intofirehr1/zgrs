package com.chinalife.etl

import java.util

import com.chinalife.util.Commons
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import scala.util.control.Breaks

class FeaturesExtraction4Month{
}

/**
  * Created by hadoop on 2017/8/8 0008.
  */
object FeaturesExtraction4Month {

  def main(args: Array[String]): Unit = {
    if(args.size < 1){
      System.err.println("params: master modelflag")
      System.exit(1)
    }
    // 设置运行参数： cpu, mem
    val conf = new SparkConf().setAppName("gs_feature_extraction_month").setMaster(args(0))

    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    //年龄（范围），性别，户口所在地 由身份证中分析获取，如不是身份证，为空，
    // 系统编号，银行代码，管理机构，转账类别，收付类型，收付项目代码，金额（范围）。
//    val dataDF = spark.sqlContext.sql("SELECT SYS_NO,BANK_CODE,MGR_BRANCH_NO,TRANS_CLASS," +
//      " MIO_CLASS,MIO_ITEM_CODE,ACC_CUST_AMNTRANG,ACC_CUST_SEX,ACC_CUST_HOME,ACC_CUST_AGERANG " +
//      "FROM ZGRS_TMP.BANK_MSG_FEAT")

    val cols = spark.table(Commons.whole_table_customs).columns.mkString("",",","").toUpperCase
    val dataDF = spark.sqlContext.sql("SELECT " + cols + " " +
        " FROM " + Commons.whole_table_customs)

//    val dataDF = spark.sqlContext.sql("SELECT " +
//              "MIOS_TRANS_CODE,SYS_NO,TRANS_CODE,TRANS_BAT_NO," +
//              "TRANS_STAT,TRANS_BAT_SEQ,PLNMIO_REC_ID,BRANCH_BANK_ACC_NO,BANK_CODE,BANK_SUB_CODE," +
//              "BANK_SUB_NAME,BANK_PROV_CODE,BANK_CITY_CODE,BANK_ACC_TYPE,BANK_ACC_NO,ACC_CUST_NAME," +
//              "BANKACC_ID_TYPE,BANKACC_ID_NO,MGR_BRANCH_NO,CNTR_NO,IPSN_NO,TRANS_CLASS,MIO_CLASS," +
//              "MIO_ITEM_CODE,PLNMIO_DATE,MIO_DATE,TRANS_AMNT,CUST_NO,GENERATE_DATE,UNITE_TRANS_CODE," +
//              "EXT01,EXT02,EXT03,EXT04,EXT05,BANK_TRANS_STAT,BANK_TRANS_DESC,MIO_CUST_NAME," +
//              "GCLK_BRANCH_NO,GCLK_CLERK_CODE,MIO_TX_NO," +
//              "ACC_CUST_SEX,ACC_CUST_HOME,ACC_CUST_AGE,ACC_CUST_AGERANG,ACC_CUST_INSTANCETYPE," +
//              "ACC_CUST_AMNTRANG,ACOUNT_TIME_RANG,ACC_CUST_MONTH,ACC_CUST_DAY,ACC_CUST_DAYRANG,KK_CNT,ACC_CUST_KM " +
//      " FROM ZGRS_TMP.CUSTOM_FEAT_KM")


    //val model_flag = args(1)
    var i : Int = 1
    val featuresMsgDF = spark.sqlContext.sql("SELECT feature_col_index, feature_col_name," +
      " feature_pcode, feature_desc FROM " + Commons.whole_table_features_msg_src + " WHERE FEATURE_FLAG=1 ")

    val featuresRangMsgCols = spark.table(Commons.whole_table_feature_rang).columns.mkString("",",","").toUpperCase
    val featuresRangMsgDF = spark.sqlContext.sql("SELECT " + featuresRangMsgCols +
      " FROM " + Commons.whole_table_feature_rang).collect()

    var dflist = new util.ArrayList[Row]()
    val featuresMsg = featuresMsgDF.rdd.collect()
    //val utils = new FeaturesExtractionUtils
    for (elem <- featuresMsg) {
      val featureNo = dataDF.rdd.map(r => parse(elem,r)).distinct.collect().zipWithIndex.toMap
      // 这块需要根据特征值，写入小类的范围或取值区间的说明
      val tmp1 = detailFeaturesCode(featureNo, elem.getAs(3), i, elem.getAs(2).toString.toInt, dflist, spark,featuresRangMsgDF)
      i = tmp1._2
      dflist = tmp1._1
    }
   println(dflist.size())

    // TODO  业务信息相关大类  21000开头

    // TODO  其他信息依次以31... 41... 等开头

    // TODO 制作特征库时，需要先看特征库里的数据是否有变动，或者直接重新生成一套特征库，但是数据一定要变动不大，之前统一的编号一定不能随意变化含义
    val featureTable = "feature_id,feature_code,feature_value,feature_name,feature_pcode,feature_ptype,feature_desc,feature_weight";
    val schemaFeatures = StructType(featureTable.split(",").map(fieldName => StructField(fieldName, StringType, true)))
    val resultDF = spark.createDataFrame(dflist, schemaFeatures)

    resultDF.createTempView("tmp_features")

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_features + " SELECT " +
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

