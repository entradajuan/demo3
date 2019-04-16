package com.softtek.demo3

import org.apache.spark.sql.SparkSession

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
    
    
  }
}