package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.clustering.StreamingKNN
import org.apache.spark.annotation.Since

object FSMLtest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val conf = new SparkConf().setMaster("local[2]").setAppName("MLStreamingTest")
    val ssc = new StreamingContext(conf, Seconds(1))

    val trainingData = ssc.textFileStream("/home/sramirez/datasets/poker-5-fold/streaming/poker-small.tra/").map(LabeledPoint.parse)
    val testData = ssc.textFileStream("/home/sramirez/datasets/poker-5-fold/streaming/poker-small.tst/").map(LabeledPoint.parse)
    
    val model = new StreamingKNN()
      .setK(1)
      .setNPartitions(2)

    model.trainOn(trainingData)
    model.predictOnValues(testData).filter(t => t._1 == t._2).count().print()
        //.transform(_.repartition(1)).saveAsTextFiles("/home/sramirez/output", "txt")
    

    ssc.start()
    ssc.awaitTermination()
  }

}