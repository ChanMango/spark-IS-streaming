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

object KNNSeqTest2 {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.
    val k = 5
    /*val points = scala.util.Random.shuffle(scala.io.Source.fromFile("/home/sramirez/datasets/poker-5-fold/streaming/part-asd.dat")
        .getLines().toSeq.map(LabeledPoint.parse))*/
    val points = scala.io.Source.fromFile("/home/sramirez/datasets/poker-5-fold/streaming/poker-medium.dat").getLines().toSeq.map(LabeledPoint.parse)
    val tree = new MTreeWrapper(points.map(lp => new DataLP(lp.features.toArray.map(_.toFloat), lp.label.toFloat)).toArray)
    val predict = (p: LabeledPoint) => {
      val neigh = tree.kNNQuery(k, p.features.toArray.map(_.toFloat)).map(t => (t.distance, t.data.asInstanceOf[DataLP]))
      val pred = neigh.map(_._2.getLabel).groupBy(identity).maxBy(_._2.size)._1
      (p.label.toFloat, pred)
    }    
    val startTime = System.currentTimeMillis()    
    println("Number of matches: " + points.map(predict).filter{ case (l,p) => l == p}.length)
    val estimatedTime = (System.currentTimeMillis() - startTime) / 1000f
    println("Estimated Time: " + estimatedTime)
    //println("Number of matches: " + preds.filter{ case (l,p) => l == p}.length)
  }

}