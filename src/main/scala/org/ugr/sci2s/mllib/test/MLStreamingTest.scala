package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature._
import org.apache.spark.annotation.Since

object MLStreamingTest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val conf = new SparkConf().setMaster("local[3]").setAppName("MLStreamingTest")
    val ssc = new StreamingContext(conf, Seconds(1))
    val k = 10

    val trainingData = ssc.textFileStream("/home/sramirez/datasets/poker-5-fold/streaming/poker-small.tra/").map(LabeledPoint.parse)
    val testData = ssc.textFileStream("/home/sramirez/datasets/poker-5-fold/streaming/poker-small.tst/").map(LabeledPoint.parse)
    
    val model = new StreamingDistributedKNN().setNTrees(4)

    model.trainOn(trainingData)
    model.predictOnValues(testData, k).filter{ case (label, pred) => label == pred}.count
        .transform(_.repartition(1)).saveAsTextFiles("/home/sramirez/output", "txt")
    

    ssc.start()
    ssc.awaitTermination()
  }

}