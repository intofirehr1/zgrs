package com.chinalife.etl

import com.chinalife.util.Commons
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

class SourceLoad{

}


/**
  * Created by hadoop on 2017/8/10 0010.
  * 原始数据加载
  */
object SourceLoad {

  def main(args: Array[String]): Unit = {

    if (args.size < 6) {
      System.err.println("params: master table")
      System.exit(1)
    }

    // 设置运行参数： cpu, mem
    val conf = new SparkConf().setAppName("zgrs_data_load").setMaster(args(0))

    val spark = SparkSession.builder().enableHiveSupport().config(conf).getOrCreate()
    //    val cal = Calendar.getInstance()
    //    val yearId = cal.get(Calendar.YEAR)
    //    val monthId = cal.get(Calendar.MONTH) + 1

    val yearId = args(3).toInt
    val monthId = args(4).toInt


    val model_flag = args(5).toInt
    if(model_flag == 0) {
      // 加载源数据
      loadSourcesMsg(spark, args, yearId, monthId)

      // 数据加载到临时表
      loadData2Tmp4Month(spark, yearId, monthId)
      //    }else {
      //      // 数据清洗时将处理状态都置为 1 成功
      //      loadData2TmpForP(spark, yearId, monthId)
      //    }
    }else {
      loadSourcesMsg4Year(spark, args, yearId, monthId)
      loadData2Tmp4Year(spark, yearId, monthId)
    }

    spark.stop()
  }

  /**
    * 加载特征配置信息
    * @param spark
    */
  def loadSourcesMsg(spark: SparkSession, args: Array[String], yearId: Int, monthId: Int): Unit = {
    spark.sqlContext.sql("use " + args(1))

//    spark.sqlContext.sql("ALTER TABLE " + args(2) +
//          " DROP IF EXISTS PARTITION(YEAR_ID=" + yearId + ", MONTH_ID=" + monthId + ")" )

    spark.sqlContext.sql("ALTER TABLE " + args(2) +
      " ADD IF NOT EXISTS PARTITION(year_id=" + yearId + ", month_id=" + monthId + ") LOCATION " +
      "'/user/hadoop/zgrs/src/" + yearId + "/" + monthId + "'")

    //    spark.sqlContext.sql("LOAD DATA LOCAL INPATH '"+ args(5)+"" +
    //                     "' OVERWRITE INTO TABLE " + args(2) +
    //                      " PARTITION(YEAR_ID="+ yearId + ", MONTH_ID=" + monthId + ")" )

  }


  /**
    * 加载特征配置信息
    * @param spark
    */
  def loadSourcesMsg4Year(spark: SparkSession, args: Array[String], yearId: Int, monthId: Int): Unit = {
    spark.sqlContext.sql("use " + args(1))

    for(month <- 1 to 12){
      spark.sqlContext.sql("ALTER TABLE " + args(2) +
        " ADD IF NOT EXISTS PARTITION(year_id=" + yearId + ", month_id=" + month + ") LOCATION " +
        "'/user/hadoop/zgrs/src/" + yearId + "/" + month + "'")
    }

  }

  /**
    * 加载数据到临时表，模型训练使用
    * @param spark
    * @param yearId
    * @param monthId
    */
  def loadData2Tmp4Month(spark: SparkSession, yearId: Int, monthId: Int): Unit = {
    //spark.sqlContext.sql("USE "+ Commons.database_tmp)
    val cols = spark.table(Commons.whole_table_sources_tmp)
      .columns.mkString("", ",", "").toUpperCase

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_sources_tmp +
      " SELECT " + cols +
      " FROM " + Commons.whole_table_sources +" WHERE year_id=" + yearId + " AND month_id=" + monthId)

  }

  /**
    * 加载数据到临时表，模型训练使用
    * @param spark
    * @param yearId
    * @param monthId
    */
  def loadData2Tmp4Year(spark: SparkSession, yearId: Int, monthId: Int): Unit = {
    //spark.sqlContext.sql("USE "+ Commons.database_tmp)
    val cols = spark.table(Commons.whole_table_sources_tmp)
      .columns.mkString("", ",", "").toUpperCase

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_sources_year_tmp +
      " SELECT " + cols +
      " FROM " + Commons.whole_table_sources +" WHERE year_id=" + yearId )
  }

  /**
    * 加载数据到临时表，并且讲trans_stat 处理状态都置为 1 表示都是成功的
    * @param spark
    * @param yearId
    * @param monthId
    */
  def loadData2TmpForP(spark: SparkSession, yearId: Int, monthId: Int): Unit = {
    //spark.sqlContext.sql("USE "+ Commons.database_tmp)
    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_sources_tmp +
      " SELECT " +
            "MIOS_TRANS_CODE,SYS_NO,TRANS_CODE,TRANS_BAT_NO,1 AS TRANS_STAT,TRANS_BAT_SEQ," +
            "PLNMIO_REC_ID,BRANCH_BANK_ACC_NO,BANK_CODE,BANK_SUB_CODE,BANK_SUB_NAME,BANK_PROV_CODE," +
            "BANK_CITY_CODE,BANK_ACC_TYPE,BANK_ACC_NO,ACC_CUST_NAME,BANKACC_ID_TYPE,BANKACC_ID_NO," +
            "MGR_BRANCH_NO,CNTR_NO,IPSN_NO,TRANS_CLASS,MIO_CLASS,MIO_ITEM_CODE,PLNMIO_DATE,MIO_DATE," +
            "TRANS_AMNT,CUST_NO,GENERATE_DATE,UNITE_TRANS_CODE,EXT01,EXT02,EXT03,EXT04,EXT05," +
            "BANK_TRANS_STAT,BANK_TRANS_DESC,MIO_CUST_NAME,GCLK_BRANCH_NO,GCLK_CLERK_CODE,MIO_TX_NO "+
      " FROM " + Commons.whole_table_sources +
      " WHERE year_id=" + yearId + " AND month_id=" + monthId)
  }
}

