package org.apache.spark.ml

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.{DefaultParamsWriter, Identifiable, MLWriter}
import org.apache.spark.sql.SQLContext
import org.json4s._
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/**
  * Created by hadoop on 2017/8/5 0005.
  */
class ParamsWriter(instance: Params) extends DefaultParamsWriter(instance: Params) {

  override protected def saveImpl(path: String): Unit = {
    ParamsWriter.saveMetadata(instance, path, sc)
  }

}

object ParamsWriter {

  /**
    *
    * @param instance
    * @param path
    * @param sc
    * @param extraMetadata
    * @param paramMap
    */
  def saveMetadata(
                    instance: Params,
                    path: String,
                    sc: SparkContext,
                    extraMetadata: Option[JObject] = None,
                    paramMap: Option[JValue] = None): Unit = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, sc, extraMetadata, paramMap)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /**
    * Helper for [[saveMetadata()]] which extracts the JSON to save.
    * This is useful for ensemble models which need to save metadata for many sub-models.
    *
    * @see [[saveMetadata()]] for details on what this includes.
    */
  def getMetadataToSave(
                         instance: Params,
                         sc: SparkContext,
                         extraMetadata: Option[JObject] = None,
                         paramMap: Option[JValue] = None): String = {
    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq.asInstanceOf[Seq[ParamPair[Any]]]
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    val metadataJson: String = compact(render(metadata))
    metadataJson
  }
}
