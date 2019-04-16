package com.softtek.demo3

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap

object ML1 {
  
   def main (args: Array[String]){
    
    val spark = SparkSession
    
      .builder()
      .appName("Spark basic example")
      .master("local[*]")
      //.master("spark://juani-VirtualBox:7077")    
      .getOrCreate()
    
    pr1(spark)
      
    
  }

  def pr1(spark: SparkSession){
    import spark.implicits._
    
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")
    
    training.show()
    
    val lr = new LogisticRegression()
    
    println(lr.explainParams())
    
    lr.setMaxIter(10)
      .setRegParam(0.01)
      
    val model1 = lr.fit(training)
    
    println(s"fit using ${model1.parent.extractParamMap()}")
    
    val paramMap1 = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter, 30)
      .put(lr.regParam -> 0.1, lr.threshold -> 0.55)

    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
    val paramMapCombined = paramMap1 ++ paramMap2
    
    val model2 = lr.fit(training, paramMapCombined)
    
    println(s"2 fit using ${model2.parent.extractParamMap()}")
    
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")      
    
    val out1 = model2.transform(test)
    
    out1.show
      
    
    
  }
}