package com.chinalife.util

import java.util

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

//import com.iss.chinalife.util.MySaveMode
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession, _}

/**
  * Created by hadoop on 2017/8/9 0009.
  */
object Commons {

  /*  database   */
  val database_src = "ZGRS_SRC"
  val database_dim = "ZGRS_DIM"
  val database_sho = "ZGRS_SHO"
  val database_tmp = "ZGRS_TMP"
  val database_feat = "ZGRS_FEAT"

  /*  table   */
  val table_sources = "SOURCES"
  val table_sources_tmp = "SOURCES_TMP"
  val table_sources_year_tmp = "SOURCES_YEAR_TMP"
  val table_customs = "CUSTOMS"
  val table_customs_year = "CUSTOMS_YEAR"
  val table_customs_pho = "CUSTOMS_PHO"
  val table_customs_year_pho = "CUSTOMS_YEAR_PHO"
  val table_customs_train = "CUSTOMS_TRAIN"
  val table_customs_train_show = "CUSTOMS_TRAIN_SHOW"
  val table_customs_year_train = "CUSTOMS_YEAR_TRAIN"
  val table_customs_year_train_show = "CUSTOMS_YEAR_TRAIN_SHOW"
  val table_features = "FEATURES"
  val table_features_year = "FEATURES_YEAR"
  val table_features_vectors = "FEATURES_VECTORS"
  val table_features_vectors_year = "FEATURES_VECTORS_YEAR"
  val table_features_msg_src = "FEATURES_MSG_SRC"
  val table_feature_rang = "FEATURES_RANG_MSG"

  /*  whole table name   */
  val whole_table_sources = database_src + "." + table_sources
  val whole_table_sources_tmp = database_tmp + "." + table_sources_tmp
  val whole_table_sources_year_tmp = database_tmp + "." + table_sources_year_tmp
  val whole_table_customs = database_tmp + "." + table_customs
  val whole_table_customs_yesr = database_tmp + "." + table_customs_year
  val whole_table_customs_pho = database_tmp + "." + table_customs_pho
  val whole_table_customs_year_pho = database_tmp + "." + table_customs_year_pho
  val whole_table_customs_train = database_sho + "." + table_customs_train
  val whole_table_customs_train_show = database_sho + "." + table_customs_train_show

  val whole_table_customs_year_train = database_sho + "." + table_customs_year_train
  val whole_table_customs_year_train_show = database_sho + "." + table_customs_year_train_show


  val whole_table_features = database_feat + "." + table_features
  val whole_table_features_year = database_feat + "." + table_features_year
  val whole_table_features_vectors = database_dim + "." + table_features_vectors
  val whole_table_features_vectors_year = database_dim + "." + table_features_vectors_year
  val whole_table_features_msg_src = database_dim + "." + table_features_msg_src
  val whole_table_feature_rang = database_dim + "." + table_feature_rang

  /* rang type */
  val AMNT_RANG = "amnt_rang"
  val AGE_RANG = "age_rang"
  val DATE_RANG = "date_rang"

  /**
    *
    * @param line
    * @return
    */
  def isColumnNameLine(line:String):Boolean = {
    if (line != null &&
      line.contains("MIOS_TRANS_CODE")) true
    else false
  }

  /**
    *
    * @param line
    * @return
    */
  def isColumnNameLine1(line:String):Boolean = {
    if (line != null &&
      line.contains("feature_col_index")) true
    else false
  }

  /**
    *
    * @param spark
    * @param dataDF
    * @param label2UserList
    * @param resultSavePath
    * @param formater
    */
  def saveResult(spark: SparkSession, dataDF : DataFrame, label2UserList: util.ArrayList[Row],
                 resultSavePath: String, formater : String): Unit ={
    val labelTable = "TRANS_CODE,TRANS_BAT_NO,PREDICT_LABEL";
    val labelSchemaFeatures = StructType(labelTable.split(",").map(fieldName => StructField(fieldName, StringType, true)))
    val label2UserDF = spark.createDataFrame(label2UserList, labelSchemaFeatures)
    val predictResult = dataDF.join(label2UserDF, Seq("TRANS_CODE","TRANS_BAT_NO"), "left")
    MySaveMode.saveMode(predictResult, resultSavePath, formater)
  }

  /**
    *
    * @param x
    * @param featureCol
    * @return
    */
  def getEncodeData4Train(x : Row, featureCol : Array[String]) = {
    var tmpArray:Array[Double] = Array[Double]()
    var featuresCnt = Array[String]()
    for (elem <- featureCol) {
      // 保存每一个特征的长度，计特征总数量,记录每个值对应的vector的索引，及每个特征的长度，之后在模型预测时，从数据里或这些数据生成新的vector
      // 特征记录，特征字段名，特征编号，特征值，特征vector大小，特征索引
      val vector = x.getAs[SparseVector](elem + "_INDEX_VEC")
      val featureVector = elem + "," + x.getAs[String](elem) + "," + "1," + (vector.toArray.size.toString) + "," + vector.indices.mkString("",",","")
      featuresCnt = featuresCnt ++ Array[String](featureVector)
      tmpArray = tmpArray ++ vector.toArray
    }
    // 保存featuresCnt 数据到
    (tmpArray,featuresCnt)
  }

  /**
    *
    * @param x
    * @param featureCol
    * @return
    */
  def getEncodeData4Model(x : Row, featureCol : Array[String], featureVectorDF: Array[Row]): Array[Double] = {
    var tmpArray:Array[Double] = Array[Double]()
    for (elem <- featureCol) {
      val featuresCode = x.getAs[String](elem)
      for (elem1 <- featureVectorDF) {
        val featureVectorCode = elem1.getAs[String]("FEATURE_COL_NAME")
        if(featureVectorCode.equals(featuresCode)){
//          println("===============  featuresCode: " + featuresCode +", add " + elem1.getAs[String]("FEATURE_VECTOR_CNT").toInt)
          tmpArray = tmpArray ++ Vectors.sparse(elem1.getAs[String]("FEATURE_VECTOR_CNT").toInt,
            Array[Int](elem1.getAs[String]("FEATURE_VECTOR_INDEX").toInt),Array[Double](1.0)).toSparse.toArray
        }
      }
    }
    tmpArray
  }

  /**
    * 判断文件目录是否存在，如果存在就删除
    * @param src_path
    * @param spark
    */
  def deleteIfExists(spark : SparkSession, src_path : String): Unit ={
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val path = new Path(src_path)

    if(hdfs.exists(path)){
      // 删除目录，包括目录下的所有数据
      hdfs.delete(path, true)
      // 为防止误删，禁止递归删除
      // hdfs.delete(path, false)
    }
  }


  /**
    * 保存特征向量属性值： 向量总大小（特征维度），当前特征值在向量中的索引位置
    * @param spark
    * @param featureVectorData
    * @return
    */
  def saveFeatureVector4Month(spark:SparkSession, featureVectorData: RDD[Row]) ={
    val featureVectorSchema = "FEATURE_CODE,FEATURE_COL_NAME,FEATURE_VALUE,FEATURE_VECTOR_CNT,FEATURE_VECTOR_INDEX"
    val festureVectorS = StructType(featureVectorSchema.split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))
    val featureVectorDF  = spark.createDataFrame(featureVectorData,festureVectorS)

    featureVectorDF.createTempView("tmp_feature_vector")

    // 去重
    val tmpFVectorDF = spark.sqlContext.sql("SELECT distinct FEATURE_COL_NAME,FEATURE_CODE, FEATURE_VALUE,FEATURE_VECTOR_CNT," +
      "FEATURE_VECTOR_INDEX " +
      " FROM tmp_feature_vector")

    tmpFVectorDF.createTempView("feature_vector")

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_features_vectors +
      " SELECT " + featureVectorDF.columns.mkString("",",","").toUpperCase + " FROM feature_vector")
  }

  /**
    * 保存特征向量属性值： 向量总大小（特征维度），当前特征值在向量中的索引位置
    * @param spark
    * @param featureVectorData
    * @return
    */
  def saveFeatureVector4Year(spark:SparkSession, featureVectorData: RDD[Row]) ={
    val featureVectorSchema = "FEATURE_CODE,FEATURE_COL_NAME,FEATURE_VALUE,FEATURE_VECTOR_CNT,FEATURE_VECTOR_INDEX"
    val festureVectorS = StructType(featureVectorSchema.split(",").map(fieldName =>
      StructField(fieldName, StringType, true)))
    val featureVectorDF  = spark.createDataFrame(featureVectorData,festureVectorS)

    featureVectorDF.createTempView("tmp_feature_vector")

    // 去重
    val tmpFVectorDF = spark.sqlContext.sql("SELECT distinct FEATURE_COL_NAME,FEATURE_CODE, FEATURE_VALUE,FEATURE_VECTOR_CNT," +
      "FEATURE_VECTOR_INDEX " +
      " FROM tmp_feature_vector")

    tmpFVectorDF.createTempView("feature_vector")

    spark.sqlContext.sql("INSERT OVERWRITE TABLE " + Commons.whole_table_features_vectors_year +
      " SELECT " + featureVectorDF.columns.mkString("",",","").toUpperCase + " FROM feature_vector")
  }

}

class Commons{
}

/**
  * class 保存用户信息和预测信息的对象
  * @param trans_code
  * @param trans_bat_no
  * @param label
  */
case class Label2User(trans_code : String, trans_bat_no : String, label : Double)

