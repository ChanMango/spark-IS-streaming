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
import org.apache.spark.mllib.feature.InstanceSelection.TreeLPNorm
import org.apache.spark.mllib.feature.StreamingDistributedKNN._
import org.apache.spark.mllib.knn.VectorWithNorm
import org.apache.spark.mllib.feature.InstanceSelection

object ISSeqTest {

  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf().setMaster("local[5]").setAppName("Simple Application")
    val sc = new SparkContext(conf)
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.
    val k = 5
    val points = scala.io.Source.fromFile("/home/sramirez/datasets/poker-5-fold/streaming/poker-medium.dat").getLines().toSeq.map(LabeledPoint.parse)   
    val tree = new MTreeWrapper(points.slice(0, 100).map(lp => new DataLP(lp.features.toArray.map(_.toFloat), lp.label.toFloat)).toArray)
    val process = (p: LabeledPoint) => {
        val neighbs = tree.kNNQuery(10, p.features.toArray.map(_.toFloat)).map{t => 
          new TreeLP(LPUtils.fromJavaLP(t.data.asInstanceOf[DataLP]), -1, NONE)
        }
        val tlp = new TreeLP(p, -1, NONE) 
        tlp -> neighbs.toArray
    } 
    for(sl <- points.slice(100, points.size).grouped(100)) {      
      val slrdd = sc.parallelize(sl.map(process), 5)    
      val filtered = InstanceSelection.instanceSelection(slrdd)  
      
      for(elem <- filtered.collect) {
        if(elem.action == INSERT)
          tree.insert(LPUtils.toJavaLP(elem.point))
        else if (elem.action == REMOVE)
          tree.remove(LPUtils.toJavaLP(elem.point))
      }  
      val hist = filtered.map(_.action).collect.groupBy(identity).map(p => p._1 -> p._2.size)
      println("Size per action " + hist)
      println("New size: " + tree.getSize)
    }
    
    val predict = (p: LabeledPoint) => {
      val neigh = tree.kNNQuery(k, p.features.toArray.map(_.toFloat)).map(t => (t.distance, t.data.asInstanceOf[DataLP]))
      val pred = neigh.map(_._2.getLabel).groupBy(identity).maxBy(_._2.size)._1
      (p.label.toFloat, pred)
    }  
    val nsize = tree.getSize()
    println("Old size: " + points.size)
    println("New size: " + nsize)
    println("Number of matches: " + points.map(predict).filter{ case (l,p) => l == p}.length)
    //val estimatedTime = (System.currentTimeMillis() - startTime) / 1000f
    //println("Estimated Time: " + estimatedTime)
  }

}