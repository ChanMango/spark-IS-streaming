package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.clustering.StreamingDistributedKNN
import org.apache.spark.annotation.Since
import xxl.core.indexStructures.mtrees.MTreeLP
import collection.JavaConversions._

object KNNSeqTest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val points = scala.util.Random.shuffle(scala.io.Source.fromFile("/home/sramirez/datasets/poker-5-fold/streaming/poker-small.tra/part-asd.dat")
        .getLines().toSeq.map(LabeledPoint.parse))
    val tree = new MTreeLP()
    points.foreach (p => tree.insert(p.features.toArray.map(_.toFloat), p.label.toFloat))    
    val predict = (p: LabeledPoint) => {
      val neigh = tree.kNNQuery(1, p.features.toArray.map(_.toFloat)).map(t => (t.getDistance.toFloat, t.getPoint.getLabel))
      val pred = neigh.map(_._2).groupBy(identity).maxBy(_._2.size)._1
      (p.label, pred)
    }    
    println("Number of matches: " + points.map(predict).filter{ case (l,p) => l == p}.length)
  }

}