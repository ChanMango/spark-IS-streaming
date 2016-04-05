package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StreamingDistributedKNN
import org.apache.spark.annotation.Since
import collection.JavaConversions._
import org.apache.spark.mllib.knn.KNNUtils
import org.apache.spark.mllib.feature.LPUtils
import org.apache.spark.mllib.feature.MTreeWrapper
import mtree.DataLP

object IndicesSeqTest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.
    val k = 1
    val points = scala.util.Random.shuffle(scala.io.Source.fromFile("/home/sramirez/datasets/poker-5-fold/streaming/part-asd.dat")
        .getLines().toSeq.map(LabeledPoint.parse))
    val tree = new MTreeWrapper(points.take(10).map(lp => new DataLP(lp.features.toArray.map(_.toFloat), lp.label.toFloat)).toArray)
    val predict = (p: LabeledPoint) => {
      tree.getIndices(LPUtils.toJavaLP(p), 1.5).map(t => (t.distance, t.data.asInstanceOf[DataLP]))
    }    
    println("Number of instances: " + points.size)
    println("Number of matches: " + points.flatMap(predict).size)
  }

}