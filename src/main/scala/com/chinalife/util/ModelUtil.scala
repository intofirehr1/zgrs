package com.chinalife.util

import java.text.DecimalFormat
import java.util

import org.apache.spark.mllib.linalg.DenseVector

/**
  * Created by hadoop on 2017/8/4 0004.
  */
object ModelUtil {


  def getPercent(denseVector: DenseVector, labels:Array[Double]) ={
    val df: DecimalFormat = new DecimalFormat("######0.00")
    val arrayList : util.ArrayList[String] = new util.ArrayList[String](denseVector.size)
    val argmax_val = denseVector(denseVector.argmax)
    println("in mygetPercent, argmax : " + argmax_val)
    var total = 1.0
    denseVector.values.foreach ( x =>
      if (x != argmax_val) total = total + math.exp(x - argmax_val)
    )
    // 存储所有类别及概率值
    for (elem <- 0 to denseVector.values.size-1) {
      arrayList.add(labels(elem) + "_" + (math.exp(denseVector(elem) - argmax_val)) / total)
    }

    val iter = arrayList.iterator()
    while(iter.hasNext){
      print(iter.next()+", ")
    }
    println("in mygetPercent , total : " + total + ", " + " arrayList size : " + arrayList.size())
    df.format((1.0/total)*100)+"%"
  }
}


