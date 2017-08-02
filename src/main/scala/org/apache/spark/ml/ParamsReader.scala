package org.apache.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.ml.util.DefaultParamsReader
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.json4s._
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/**
  * Created by hadoop on 2017/8/5 0005.
  */
class ParamsReader[T] extends DefaultParamsReader[T] {

}

object ParamsReader {

  case class Metadata(className: String,
                      uid: String,
                      timestamp: Long,
                      sparkVersion: String,
                      params: JValue,
                      metadata: JValue,
                      metadataJson: String ) {

    /**
      * Get the JSON value of the [[org.apache.spark.ml.param.Param]] of the given name.
      * This can be useful for getting a Param value before an instance of `Params`
      * is available.
      */
    def getParamValue(paramName: String): JValue = {
      implicit val format = DefaultFormats
      params match {
        case JObject(pairs) =>
          val values = pairs.filter { case (pName, jsonValue) =>
            pName == paramName
          }.map(_._2)
          assert(values.length == 1, s"Expected one instance of Param '$paramName' but found" +
            s" ${values.length} in JSON Params: " + pairs.map(_.toString).mkString(", "))
          values.head
        case _ =>
          throw new IllegalArgumentException(
            s"Cannot recognize JSON metadata: $metadataJson.")
      }
    }
  }

  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    val metadata = DefaultParamsReader.loadMetadata(path,sc,expectedClassName)

    new Metadata(metadata.className,metadata.uid,
      metadata.timestamp,metadata.sparkVersion,
      metadata.params, metadata.metadata,
      metadata.metadataJson)
  }




}
