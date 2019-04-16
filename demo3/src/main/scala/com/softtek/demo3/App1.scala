package com.softtek.demo3

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructType

object App1 {
  
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
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    
    val fields = "texto".split(" ").map(fieldName => StructField(fieldName, StringType, nullable=true))
    val schema = StructType(fields)

    //val linesDF = spark.read.option("header", "false").schema(schema).csv("D:/data/wordsCount.txt")
    
    val linesDF = spark.read.option("header", "false").schema(schema).csv("/home/hduser/Documents/spark/toRun/data/wordsCount.txt")
    
    val linesDS = linesDF.as[String] 
    
    val wordsDS = linesDS.flatMap(_.split(" ")).map((_, 1))
    wordsDS.show()
    wordsDS.printSchema()
    
    
    val newNames = Seq("id1", "id2")
    
    import org.apache.spark.sql.functions._ 

    val countedDS = wordsDS
                    .groupByKey(_._1)
                    .reduceGroups((a,b) => (a._1, a._2+b._2))
                    //.toDF(newNames: _*)
                    //.select("id2")
                    .map(_._2).sort(desc("_2"))
                    //.select($"_2")
                    
                    
    countedDS.show()
    countedDS.printSchema()
    
    
    
    
  }


}