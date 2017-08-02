package com.chinalife.etl

import com.chinalife.util.Commons
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

import scala.util.control.Breaks

class YearTrainDataSho {

}

/**
  * Created by hadoop on 2017/8/1 0001.
  */
object YearTrainDataSho {

  def main(args: Array[String]): Unit = {
    // 设置运行参数： cpu, mem
    val conf = new SparkConf().setAppName("gs_data_show_year").setMaster(args(0))

    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()

    val yearId = args(1).toInt
    val monthId = args(2).toInt

//    val dataDF = spark.sqlContext.sql("SELECT " + spark.sqlContext.table(Commons.whole_table_customs_year_train).
//      columns.mkString("",",","").toUpperCase + " FROM " + Commons.whole_table_customs_year_train )

    val dataDF = spark.sqlContext.sql("SELECT MIOS_TRANS_CODE,SYS_NO,TRANS_CODE,TRANS_BAT_NO,TRANS_STAT," +
      "ACC_CUST_NAME,BANKACC_ID_TYPE,BANKACC_ID_NO,FEATURES,TRAIN_LABEL,TRAIN_ACCURACY,ACC_CUST_SEX," +
      "ACC_CUST_HOME,ACC_CUST_AGE,ACC_CUST_AGERANG,ACC_CUST_INSTANCETYPE," +
      "ACC_CUST_AMNTRANG,ACC_CUST_MONTH,ACC_CUST_DAY,ACC_CUST_DAYRANG,KK_CNT," +
      "ACC_CUST_KM " +
      "FROM " + Commons.whole_table_customs_year_train)

    val featuresDF = spark.sqlContext.sql("SELECT feature_id,feature_code,feature_value," +
      "feature_name, feature_pcode,feature_ptype,feature_desc,feature_weight" +
      " FROM " + Commons.whole_table_features_year).rdd.collect()

    // dataDF 需要遍历，然后形成新的df，然后存入数据库，
    //val utils = new TrainDataUtils
    val newDataDF = dataDF.rdd.map { row =>
      val features = row.getAs[String]("FEATURES").split("_")
      var featuesDesc = ""
      for(elem <- features){
        featuesDesc = featuesDesc + " ;" + getFeaturesValue(elem, featuresDF)
      }
      Row.fromSeq(row.toSeq ++ Seq(featuesDesc))
    }

    val schema = dataDF.columns.mkString("",",","") ++ ",FEATURES_DESC"
    val schemaFeatures = StructType(schema.split(",").map(fieldName => StructField(fieldName, StringType, true)))
    val showDF  = spark.createDataFrame(newDataDF, schemaFeatures)
    showDF.createTempView("tmp_sho")

    // 插入数据，需要注意年模型预测完之后的数据是怎么保存，以年为分区是否过大，而且如果是每一次一个模型数据输入的话，
    // 最造成数据覆盖，所以这里保存数据，需要明确设置数据保存分区，不能造成数据覆盖

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_customs_year_train_show+ " " +
      " PARTITION (year_id=" + yearId + ", month_id=" + monthId + ")" +
      " SELECT " + schema + " FROM tmp_sho " )

    spark.stop()

  }

  /**
    *
    * @param code
    * @param featruesArray
    * @return
    */
  def getFeaturesValue(code : String, featruesArray: Array[Row]): String = {
    var featuresValue = ""
    val loop = new Breaks
    loop.breakable {
      for (features <- featruesArray) {
        if (code.equals(features.getAs("feature_code"))) {
          featuresValue = features.getAs("feature_ptype").toString + ": " + features.getAs("feature_desc").toString
          loop.break
        }
      }
    }
    featuresValue
  }
}

