package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.clustering.StreamingKMeans

object FSMLtest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val conf = new SparkConf().setMaster("local[2]").setAppName("MLStreamingTest")
    val ssc = new StreamingContext(conf, Seconds(1))

    val trainingData = ssc.textFileStream("/training/data/dir").map(Vectors.parse)
    val testData = ssc.textFileStream("/testing/data/dir").map(LabeledPoint.parse)

    val numDimensions = 3
    val numClusters = 2
    val model = new StreamingKMeans()
      .setK(numClusters)
      .setDecayFactor(1.0)
      .setRandomCenters(numDimensions, 0.0)

    model.trainOn(trainingData)
    model.predictOnValues(testData.map(lp => (lp.label, lp.features))).print()

    ssc.start()
    ssc.awaitTermination()
  }

}