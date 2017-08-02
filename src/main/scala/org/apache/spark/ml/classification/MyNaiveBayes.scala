
package org.apache.spark.ml.classification

import java.util

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{Vector, _}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset, Row}

  /*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



  /**
    * Params for Naive Bayes Classifiers.
    */
  private[classification] trait NaiveBayesParams extends PredictorParams with HasWeightCol {

    /**
      * The smoothing parameter.
      * (default = 1.0).
      * @group param
      */
    final val smoothing: DoubleParam = new DoubleParam(this, "smoothing", "The smoothing parameter.",
      ParamValidators.gtEq(0))

    /** @group getParam */
    final def getSmoothing: Double = $(smoothing)

    /**
      * The model type which is a string (case-sensitive).
      * Supported options: "multinomial" and "bernoulli".
      * (default = multinomial)
      * @group param
      */
    final val modelType: Param[String] = new Param[String](this, "modelType", "The model type " +
      "which is a string (case-sensitive). Supported options: multinomial (default) and bernoulli.",
      ParamValidators.inArray[String](NaiveBayes.supportedModelTypes.toArray))

    /** @group getParam */
    final def getModelType: String = $(modelType)
  }

  // scalastyle:off line.size.limit
  /**
    * Naive Bayes Classifiers.
    * It supports Multinomial NB
    * (see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html">
    * here</a>)
    * which can handle finitely supported discrete data. For example, by converting documents into
    * TF-IDF vectors, it can be used for document classification. By making every vector a
    * binary (0/1) data, it can also be used as Bernoulli NB
    * (see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html">
    * here</a>).
    * The input feature values must be nonnegative.
    */
  // scalastyle:on line.size.limit

  class MyNaiveBayes @Since("1.5.0") (
                                     @Since("1.5.0") override val uid: String)
    extends ProbabilisticClassifier[Vector, MyNaiveBayes, MyNaiveBayesModel]
      with NaiveBayesParams with DefaultParamsWritable {

    import org.apache.spark.mllib.classification.MyNaiveBayes._

    @Since("1.5.0")
    def this() = this(Identifiable.randomUID("nb"))

    /**
      * Set the smoothing parameter.
      * Default is 1.0.
      * @group setParam
      */
    @Since("1.5.0")
    def setSmoothing(value: Double): this.type = set(smoothing, value)
    setDefault(smoothing -> 1.0)

    /**
      * Set the model type using a string (case-sensitive).
      * Supported options: "multinomial" and "bernoulli".
      * Default is "multinomial"
      * @group setParam
      */
    @Since("1.5.0")
    def setModelType(value: String): this.type = set(modelType, value)
    setDefault(modelType -> MyNaiveBayes.Multinomial)

    /**
      * Sets the value of param [[weightCol]].
      * If this is not set or empty, we treat all instance weights as 1.0.
      * Default is not set, so all instances have weight one.
      *
      * @group setParam
      */
    @Since("2.1.0")
    def setWeightCol(value: String): this.type = set(weightCol, value)

    override protected def train(dataset: Dataset[_]): MyNaiveBayesModel = {
      val labelcnt = dataset.groupBy(${labelCol}).count().collect().map(x => (x.getDouble(0),x.getLong(1)))
      val labelWeight = getUnPerent(labelcnt)
      // TODO 需要做一个可配置功能，可通过设置修改各类别样本的权重，并且做归一化处理
      trainWithLabelCheck1(dataset, positiveLabel = true,  labelWeight)

//      trainWithLabelCheck(dataset, positiveLabel = true)
    }

//    override protected def train(dataset: Dataset[_], labelWeight: util.HashMap[Double,Double]): MyNaiveBayesModel = {
//      trainWithLabelCheck1(dataset, positiveLabel = true,  labelWeight)
//    }

    /**
      * ml assumes input labels in range [0, numClasses). But this implementation
      * is also called by mllib NaiveBayes which allows other kinds of input labels
      * such as {-1, +1}. `positiveLabel` is used to determine whether the label
      * should be checked and it should be removed when we remove mllib NaiveBayes.
      */
    private[spark] def trainWithLabelCheck(
                                            dataset: Dataset[_],
                                            positiveLabel: Boolean): MyNaiveBayesModel = {
      println(" ---------  in myNaiveBayes : positiveLabel: " + positiveLabel + ",thresholds: " + thresholds)
      if (positiveLabel && isDefined(thresholds)) {
        val numClasses = getNumClasses(dataset)
        require($(thresholds).length == numClasses, this.getClass.getSimpleName +
          ".train() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
      }

      val modelTypeValue = $(modelType)
      println(" ---------  in myNaiveBayes :modelTypeValue: " + modelTypeValue)
      val requireValues: Vector => Unit = {
        modelTypeValue match {
          case Multinomial =>
            MyNaiveBayes.requireNonnegativeValues
          case Bernoulli =>
            MyNaiveBayes.requireZeroOneBernoulliValues
          case _ =>
            // This should never happen.
            throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
        }
      }

      val instr = Instrumentation.create(this, dataset)
      instr.logParams(labelCol, featuresCol, weightCol, predictionCol, rawPredictionCol,
        probabilityCol, modelType, smoothing, thresholds)

      val numFeatures = dataset.select(col($(featuresCol))).head().getAs[Vector](0).size
      instr.logNumFeatures(numFeatures)
      val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

      println(" ---------  in myNaiveBayes :weightCol: " + w)
      // Aggregates term frequencies per label.
      // TODO: Calling aggregateByKey and collect creates two stages, we can implement something
      // TODO: similar to reduceByKeyLocally to save one stage.
      val aggregated = dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd
        .map { row => (row.getDouble(0), (row.getDouble(1), row.getAs[Vector](2)))
        }.aggregateByKey[(Double, DenseVector)]((0.0, Vectors.zeros(numFeatures).toDense))(
        seqOp = {
          case ((weightSum: Double, featureSum: DenseVector), (weight, features)) =>
            requireValues(features)
            BLAS.axpy(weight, features, featureSum)
            (weightSum + weight, featureSum)
        },
        combOp = {
          case ((weightSum1, featureSum1), (weightSum2, featureSum2)) =>
            BLAS.axpy(1.0, featureSum2, featureSum1)
            (weightSum1 + weightSum2, featureSum1)
        }).collect().sortBy(_._1)

      val numLabels = aggregated.length
      println(" ---------  in myNaiveBayes :numLabels: " + numLabels)
      instr.logNumClasses(numLabels)
      val numDocuments = aggregated.map(_._2._1).sum
      println(" ---------  in myNaiveBayes :numDocuments: " + numDocuments)
      val labelArray = new Array[Double](numLabels)
      val piArray = new Array[Double](numLabels)
      val thetaArray = new Array[Double](numLabels * numFeatures)

      val lambda = $(smoothing)
      val piLogDenom = math.log(numDocuments + numLabels * lambda)
      var i = 0
      aggregated.foreach { case (label, (n, sumTermFreqs)) =>
        labelArray(i) = label
        piArray(i) = math.log(n + lambda) - piLogDenom
        val thetaLogDenom = $(modelType) match {
          case Multinomial => math.log(sumTermFreqs.values.sum + numFeatures * lambda)
          case Bernoulli => math.log(n + 2.0 * lambda)
          case _ =>
            // This should never happen.
            throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
        }
        var j = 0
        while (j < numFeatures) {
          thetaArray(i * numFeatures + j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
          j += 1
        }
        i += 1
      }

      val pi = Vectors.dense(piArray)
      val theta = new DenseMatrix(numLabels, numFeatures, thetaArray, true)
      val model = new MyNaiveBayesModel(uid, pi, theta).setOldLabels(labelArray)
      instr.logSuccess(model)
      model
    }


    /**
      *
      * @param labelcnt
      * @return
      */
    def getUnPerent(labelcnt: Array[(Double,Long)]):  java.util.HashMap[Double,Double] = {
      val labelWeight = new java.util.HashMap[Double,Double]
      val tmpWeight = new java.util.HashMap[Double,Double]
      var total = 0.0
      for (elem <- labelcnt) {
        total = total + elem._2
      }

      if(labelcnt.size==2){
        for(elem <- labelcnt){
          labelWeight.put(elem._1, 1-(elem._2/total))
        }
      }else {
        var lam = 10.0
        var tmpTotal = 0.0
        for(elem <- labelcnt){
          println(elem._1 + ", lable; " + total + ", total; " + ((elem._2/total))+", elem/total;"  + elem._2 + ", percent: " + ((1-(elem._2/total))/(elem._2/total)) + " , ======= count")
          val tp = ((1-(elem._2/total))/(elem._2/total))
          tmpWeight.put(elem._1, tp)
          tmpTotal = tmpTotal + tp
        }
        println( tmpTotal + ", ======= count tmpTotal")

        for (elem <- labelcnt) {
          println(elem._1 + ", lable; " + total + ", total; " + ((elem._2/total))+", elem/total;"  + elem._2 + ", percent: " +  tmpWeight.get(elem._1)/tmpTotal + " , ======= count")
          labelWeight.put(elem._1, tmpWeight.get(elem._1)/tmpTotal)
        }
      }
      labelWeight
    }


    /**
      * ml assumes input labels in range [0, numClasses). But this implementation
      * is also called by mllib NaiveBayes which allows other kinds of input labels
      * such as {-1, +1}. `positiveLabel` is used to determine whether the label
      * should be checked and it should be removed when we remove mllib NaiveBayes.
      */
    private[spark] def trainWithLabelCheck1(
                                            dataset: Dataset[_],
                                            positiveLabel: Boolean, labelWeight: util.HashMap[Double,Double]): MyNaiveBayesModel = {
      println(" ---------  in myNaiveBayes : positiveLabel: " + positiveLabel + ",thresholds: " + thresholds)
      if (positiveLabel && isDefined(thresholds)) {
        val numClasses = getNumClasses(dataset)
        require($(thresholds).length == numClasses, this.getClass.getSimpleName +
          ".train() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
      }

      val modelTypeValue = $(modelType)
      println(" ---------  in myNaiveBayes :modelTypeValue: " + modelTypeValue)
      val requireValues: Vector => Unit = {
        modelTypeValue match {
          case Multinomial =>
            MyNaiveBayes.requireNonnegativeValues
          case Bernoulli =>
            MyNaiveBayes.requireZeroOneBernoulliValues
          case _ =>
            // This should never happen.
            throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
        }
      }

      val instr = Instrumentation.create(this, dataset)
      instr.logParams(labelCol, featuresCol, weightCol, predictionCol, rawPredictionCol,
        probabilityCol, modelType, smoothing, thresholds)

      val numFeatures = dataset.select(col($(featuresCol))).head().getAs[Vector](0).size
      instr.logNumFeatures(numFeatures)
      val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

      println(" ---------  in myNaiveBayes :weightCol: " + w)
      // Aggregates term frequencies per label.
      // TODO: Calling aggregateByKey and collect creates two stages, we can implement something
      // TODO: similar to reduceByKeyLocally to save one stage.
      val aggregated = dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd
        .map { row => (row.getDouble(0), (row.getDouble(1), row.getAs[Vector](2)))
        }.aggregateByKey[(Double, DenseVector)]((0.0, Vectors.zeros(numFeatures).toDense))(
        seqOp = {
          case ((weightSum: Double, featureSum: DenseVector), (weight, features)) =>
            requireValues(features)
            BLAS.axpy(weight, features, featureSum)
            (weightSum + weight, featureSum)
        },
        combOp = {
          case ((weightSum1, featureSum1), (weightSum2, featureSum2)) =>
            BLAS.axpy(1.0, featureSum2, featureSum1)
            (weightSum1 + weightSum2, featureSum1)
        }).collect().sortBy(_._1)

      val numLabels = aggregated.length
      println(" ---------  in myNaiveBayes :numLabels: " + numLabels)
      instr.logNumClasses(numLabels)
      val numDocuments = aggregated.map(_._2._1).sum
      println(" ---------  in myNaiveBayes :numDocuments: " + numDocuments)
      val labelArray = new Array[Double](numLabels)
      val piArray = new Array[Double](numLabels)
      val thetaArray = new Array[Double](numLabels * numFeatures)

      val lambda = $(smoothing)
      val piLogDenom = math.log(numDocuments + numLabels * lambda)
      var i = 0
      aggregated.foreach { case (label, (n, sumTermFreqs)) =>
        labelArray(i) = label
        val labWeight = labelWeight.get(label)
        piArray(i) = math.log(n + lambda) - piLogDenom + math.log(labWeight)
        val thetaLogDenom = $(modelType) match {
          case Multinomial => math.log(sumTermFreqs.values.sum + numFeatures * lambda)
          case Bernoulli => math.log(n + 2.0 * lambda)
          case _ =>
            // This should never happen.
            throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
        }
        var j = 0
        while (j < numFeatures) {
          thetaArray(i * numFeatures + j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
          j += 1
        }
        i += 1
      }

      val pi = Vectors.dense(piArray)
      val theta = new DenseMatrix(numLabels, numFeatures, thetaArray, true)
      val model = new MyNaiveBayesModel(uid, pi, theta).setOldLabels(labelArray)
      instr.logSuccess(model)
      model
    }


    @Since("1.5.0")
    override def copy(extra: ParamMap): MyNaiveBayes = defaultCopy(extra)
  }

  @Since("1.6.0")
  object MyNaiveBayes extends DefaultParamsReadable[MyNaiveBayes] {
    /** String name for multinomial model type. */
    private[classification] val Multinomial: String = "multinomial"

    /** String name for Bernoulli model type. */
    private[classification] val Bernoulli: String = "bernoulli"

    /* Set of modelTypes that NaiveBayes supports */
    private[classification] val supportedModelTypes = Set(Multinomial, Bernoulli)

    private[MyNaiveBayes] def requireNonnegativeValues(v: Vector): Unit = {
     // println(" ---------  in myNaiveBayes requireNonnegativeValues :v: " + v.toArray.mkString("",",",""))
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }

      require(values.forall(_ >= 0.0),
        s"Naive Bayes requires nonnegative feature values but found $v.")
    }

    private[MyNaiveBayes] def requireZeroOneBernoulliValues(v: Vector): Unit = {
      //println(" ---------  in myNaiveBayes requireZeroOneBernoulliValues :v: " + v.toArray.mkString("",",",""))
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }

      require(values.forall(v => v == 0.0 || v == 1.0),
        s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
    }

    @Since("1.6.0")
    override def load(path: String): MyNaiveBayes = super.load(path)
  }

  /**
    * Model produced by [[NaiveBayes]]
    * @param pi log of class priors, whose dimension is C (number of classes)
    * @param theta log of class conditional probabilities, whose dimension is C (number of classes)
    *              by D (number of features)
    */
  @Since("1.5.0")
  class MyNaiveBayesModel private[ml] (
                                      @Since("1.5.0") override val uid: String,
                                      @Since("2.0.0") val pi: Vector,
                                      @Since("2.0.0") val theta: Matrix)
    extends ProbabilisticClassificationModel[Vector, MyNaiveBayesModel]
      with NaiveBayesParams with MLWritable {

    import MyNaiveBayes.{Bernoulli, Multinomial}

    /**
      * mllib NaiveBayes is a wrapper of ml implementation currently.
      * Input labels of mllib could be {-1, +1} and mllib NaiveBayesModel exposes labels,
      * both of which are different from ml, so we should store the labels sequentially
      * to be called by mllib. This should be removed when we remove mllib NaiveBayes.
      */
    private[spark] var oldLabels: Array[Double] = null

    private[spark] def setOldLabels(labels: Array[Double]): this.type = {
      this.oldLabels = labels
      this
    }

    /**
      * Bernoulli scoring requires log(condprob) if 1, log(1-condprob) if 0.
      * This precomputes log(1.0 - exp(theta)) and its sum which are used for the linear algebra
      * application of this condition (in predict function).
      */
    private lazy val (thetaMinusNegTheta, negThetaSum) = $(modelType) match {
      case Multinomial => (None, None)
      case Bernoulli =>
        val negTheta = theta.map(value => math.log(1.0 - math.exp(value)))
        val ones = new DenseVector(Array.fill(theta.numCols) {1.0})
        val thetaMinusNegTheta = theta.map { value =>
          value - math.log(1.0 - math.exp(value))
        }
        (Option(thetaMinusNegTheta), Option(negTheta.multiply(ones)))
      case _ =>
        // This should never happen.
        throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
    }

    @Since("1.6.0")
    override val numFeatures: Int = theta.numCols

    @Since("1.5.0")
    override val numClasses: Int = pi.size

    private def multinomialCalculation(features: Vector) = {
//      println(" ---------  in myNaiveBayes multinomialCalculation : features: " + features.toArray.mkString("",",",""))
      val prob = theta.multiply(features)
      BLAS.axpy(1.0, pi, prob)
      println(" ---------  in myNaiveBayes multinomialCalculation : prob: " + prob.values.mkString("",",",""))
      prob
    }

    private def bernoulliCalculation(features: Vector) = {
//      println(" ---------  in myNaiveBayes bernoulliCalculation : features: " + features.toArray.mkString("",",",""))
      features.foreachActive((_, value) =>
        require(value == 0.0 || value == 1.0,
          s"Bernoulli naive Bayes requires 0 or 1 feature values but found $features.")
      )
      val prob = thetaMinusNegTheta.get.multiply(features)
      BLAS.axpy(1.0, pi, prob)
      BLAS.axpy(1.0, negThetaSum.get, prob)
      println(" ---------  in myNaiveBayes bernoulliCalculation : prob: " + prob.values.mkString("",",",""))
      prob
    }

    override protected def predictRaw(features: Vector): Vector = {
      println(" ---------  in myNaiveBayes predictRaw : modelType: " + $(modelType))
      $(modelType) match {
        case Multinomial =>
          multinomialCalculation(features)
        case Bernoulli =>
          bernoulliCalculation(features)
        case _ =>
          // This should never happen.
          throw new UnknownError(s"Invalid modelType: ${$(modelType)}.")
      }
    }

    override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
      rawPrediction match {
        case dv: DenseVector =>
          var i = 0
          val size = dv.size
          val maxLog = dv.values.max
          while (i < size) {
            dv.values(i) = math.exp(dv.values(i) - maxLog)
            i += 1
          }
          val probSum = dv.values.sum
          i = 0
          while (i < size) {
            dv.values(i) = dv.values(i) / probSum
            i += 1
          }
          dv
        case sv: SparseVector =>
          throw new RuntimeException("Unexpected error in NaiveBayesModel:" +
            " raw2probabilityInPlace encountered SparseVector")
      }
    }

    @Since("1.5.0")
    override def copy(extra: ParamMap): MyNaiveBayesModel = {
      copyValues(new MyNaiveBayesModel(uid, pi, theta).setParent(this.parent), extra)
    }

    @Since("1.5.0")
    override def toString: String = {
      s"NaiveBayesModel (uid=$uid) with ${pi.size} classes"
    }

    @Since("1.6.0")
    override def write: MLWriter = new MyNaiveBayesModel.MyNaiveBayesModelWriter(this)
  }

  @Since("1.6.0")
  object MyNaiveBayesModel extends MLReadable[MyNaiveBayesModel] {

    @Since("1.6.0")
    override def read: MLReader[MyNaiveBayesModel] = new MyNaiveBayesModelReader

    @Since("1.6.0")
    override def load(path: String): MyNaiveBayesModel = super.load(path)

    /** [[MLWriter]] instance for [[NaiveBayesModel]] */
    private[MyNaiveBayesModel] class MyNaiveBayesModelWriter(instance: MyNaiveBayesModel) extends MLWriter {

      private case class Data(pi: Vector, theta: Matrix)

      override protected def saveImpl(path: String): Unit = {
        // Save metadata and Params
        DefaultParamsWriter.saveMetadata(instance, path, sc)
        // Save model data: pi, theta
        val data = Data(instance.pi, instance.theta)
        val dataPath = new Path(path, "data").toString
        sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
      }
    }

    private class MyNaiveBayesModelReader extends MLReader[MyNaiveBayesModel] {

      /** Checked against metadata when loading model */
      private val className = classOf[MyNaiveBayesModel].getName

      override def load(path: String): MyNaiveBayesModel = {
        val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

        val dataPath = new Path(path, "data").toString
        val data = sparkSession.read.parquet(dataPath)
        val vecConverted = MLUtils.convertVectorColumnsToML(data, "pi")
        val Row(pi: Vector, theta: Matrix) = MLUtils.convertMatrixColumnsToML(vecConverted, "theta")
          .select("pi", "theta")
          .head()
        val model = new MyNaiveBayesModel(metadata.uid, pi, theta)

        DefaultParamsReader.getAndSetParams(model, metadata)
        model
      }
    }
  }


