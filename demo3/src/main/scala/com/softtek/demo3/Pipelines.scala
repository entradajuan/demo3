package com.softtek.demo3

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel

object Pipelines {
 
  def main(args: Array[String]){
    
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
    
    val training =  spark.createDataFrame(Seq(
      (0L, "a b c spark d e", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "78 f f s ce", 0.0),
      (5L, "3e3e f t spark 44", 1.0),
      (6L, "ed Spark", 1.0),
      (6L, "qwed 000 Spark qwefqw", 1.0),
      (6L, "qwe d Spark fvdv", 1.0),
      (6L, "55 55 55 5 5 Spark 12123 32123", 1.0),
      (6L, "y45 y54 y654y Spark 12312  edewf", 1.0),
      (6L, "Spark33 qwef qwef ", 1.0),
      (6L, "Spark84 51  48 512 ", 1.0),
      (6L, "qwef qwef Spark qwef qwef qwef", 1.0),
      (6L, "Alemania", 0.0),
      (6L, "Belgica", 0.0),
      (6L, "Notre dame 2000", 0.0),
      (6L, "UUSSSS S SS SS ", 0.0),
      (6L, "Pescato asd Bar", 0.0)      
    )).toDF("id", "text", "label")
    
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
      
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
      
    val pipe = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))
      
    val model = pipe.fit(training)
    
    model.write.overwrite().save("D:/data/models/lr1Model") 
    pipe.write.overwrite().save("D:/data/models/lr1Pipe")
    
    val sameModel = PipelineModel.load("D:/data/models/lr1Model")

    val test = spark.createDataFrame(Seq(
        (100L, "adgfh asdf asdf"),
        (101L, "asdf spark 333"),
        (102L, "2 5 spark")
      )).toDF("id", "text")
      
    val out = sameModel.transform(test)
    
    out.show()
    
    
    
    
    
  }
}