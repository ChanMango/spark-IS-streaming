package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StreamingDistributedKNN
import org.apache.spark.annotation.Since
import xxl.core.indexStructures.mtrees.MTreeLP
import collection.JavaConversions._

object OverlapSeqTest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val points = scala.util.Random.shuffle(scala.io.Source.fromFile("/home/sramirez/datasets/poker-5-fold/streaming/poker-small.tra/part-asd.dat")
        .getLines().toSeq.map(LabeledPoint.parse))
    val tree = new MTreeLP()
    val firstLoad = points.take(10)
    firstLoad.foreach (p => tree.insert(p.features.toArray.map(_.toFloat), p.label.toFloat))  
    
    val nindices = points.map{ p => 
      val nearest = tree.kNNQuery(1, points.seq(0).features.toArray.map(_.toFloat)).map(t => (t.getDistance.toFloat, t.getPoint)).get(0)
      tree.overlapQuery(nearest._2.getFeatures, 5.5).length
    }
    
    println("Number of matches in range: " + nindices.filter(_ != 1).length)
  }

}