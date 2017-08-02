package com.chinalife.classification

import org.apache.spark.ml.param._

/**
  * Created by hadoop on 2017/8/3 0003.
  */
trait AdaboostNaiveBayesParams extends Params {

  // 进行adaboost时的最高迭代次数
  final val maxIter: IntParam = new IntParam(this, "maxIter", "max number of iterations")
    def getMaxIter: Int = $(maxIter)

  // 进行adaboost时准确率变化小于某个阈值时迭代提前终止
  final val threshold: DoubleParam = new DoubleParam(this, "threshold", "improvement threshold among iterations")
    def getThreshold: Double = $(threshold)

  // 朴素Bayes的平滑系数
  final val smoothing : DoubleParam = new DoubleParam(this, "smoothing", "naive bayes smooth")
    def getSmoothing : Double = $(smoothing)

   // 朴素Bayes类型"multinomial"(default) and "bernoulli"
  final val modelType : Param[String] = new Param[String](this, "modelType", "naive bayes model type")
    def getModelType : String = $(modelType)

  // 朴素Bayes 模型地址 有可能是从1-12月这样的，也有可能都是相同的
  final val modelPath : StringArrayParam = new StringArrayParam(this, "modelPath", "naive bayes model path")
    def getModelPath : Array[String] = $(modelPath)

  final val customWeights : DoubleArrayParam = new DoubleArrayParam(this, "customWeights", "month model weights")
    def getCustomWeights : Array[Double] = $(customWeights)

  /**
    * Param for raw prediction (a.k.a. confidence) column name.
    * @group param
    */
  final val prob: Param[String] = new Param[String](this, "porb", "naive bayes predict prob")

  setDefault(prob, "prob")

  /** @group getParam */
  final def getProb: String = $(prob)

}
