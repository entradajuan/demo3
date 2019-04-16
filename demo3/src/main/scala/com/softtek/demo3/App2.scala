package com.softtek.demo3

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.Row
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.Summarizer._


object App2 {
    
  def main (args: Array[String]){
    
    val spark = SparkSession
      .builder()
      .appName("Spark basic example")
      .master("local[*]")
      //.master("spark://juani-VirtualBox:7077")    
      .getOrCreate()
    
    pr3(spark)
      
    
  }

  def pr1 (spark: SparkSession) {
    
    import spark.implicits._
    
    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )    

    
    val df = data.map(Tuple1.apply).toDF("features")
    
    df.show()
    val Row(coeff1:Matrix) = Correlation.corr(df, "features").head()
    
    println(s"Pearson corr ma: \n $coeff1")
    
    val Row(coeff2:Matrix) = Correlation.corr(df, "features", "spearman").head()
    println(s"\n $coeff2")
    
    
    
  }
  
  def pr2(spark: SparkSession) {
    import spark.implicits._
        
    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )    

    val df = data.toDF("label", "features")
    val chi = ChiSquareTest.test(df, "features", "label").head()
    
    println(s"pValues = ${chi.getAs[Vector](0)}")
    println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
    println(s"statistics ${chi.getAs[Vector](2)}")
  
  
  
  }
  
  def pr3(spark: SparkSession){
    import spark.implicits._
    
    val data = Seq(
      (Vectors.dense(2.0, 3.0, 5.0), 1.0),
      (Vectors.dense(4.0, 6.0, 7.0), 2.0)
    )
    
    val df = data.toDF("features", "weight")
    
    val (meanVal, varianceVal) = df.select(metrics("mean", "variance")
      .summary($"features", $"weight").as("summary"))
      .select("summary.mean", "summary.variance")
      .as[(Vector, Vector)].first()
    
    println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
    
    val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
      .as[(Vector, Vector)].first()
    
    println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
        
  }
  
}