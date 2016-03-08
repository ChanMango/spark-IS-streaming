package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.clustering.StreamingKMeans
import org.apache.spark.mllib.clustering.StreamingKNNModel
import org.apache.spark.mllib.clustering.StreamingKNN

object FSMLtest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val conf = new SparkConf().setMaster("local[2]").setAppName("MLStreamingTest")
    val ssc = new StreamingContext(conf, Seconds(1))

    val trainingData = ssc.textFileStream("/training/data/dir").map(Vectors.parse)
    val testData = ssc.textFileStream("/testing/data/dir").map(LabeledPoint.parse)

    val model = new StreamingKNN()
      .setK(5)
      .setNPartitions(3)

    model.trainOn(trainingData)
    model.predictOnValues(testData.map(lp => (lp.label, lp.features))).print()

    ssc.start()
    ssc.awaitTermination()
  }

}