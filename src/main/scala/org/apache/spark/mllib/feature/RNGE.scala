package org.apache.spark.mllib.feature

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{StreamingDistributedKNN => SDK}
import org.apache.spark.mllib.knn.VectorWithNorm
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

/**
 * @author sramirez
 */



object RNGE {
  
  def edition(neighbors: RDD[(LabeledPoint, Array[(LabeledPoint, Int)])], secondOrder: Boolean = false) = {
    
    val edited = neighbors.map{ case (point, neigs) => 
      
      val lpnorms = (point.label, new VectorWithNorm(point.features), -1) +:
        neigs.map(n => (n._1.label, new VectorWithNorm(n._1.features), n._2))
        
      val graph = Array.fill[Boolean](lpnorms.length, lpnorms.length)(true)
      
      /* Compute the relative neighbor graph for edition (RNG-E) */
      for(i <- 0 until lpnorms.length) {
        for(j <- 0 until lpnorms.length) {
          val dij = lpnorms(i)._2.fastDistance(lpnorms(j)._2)
          for(k <- 0 until lpnorms.length if k != i && k != j) {
            if(dij > math.max(lpnorms(i)._2.fastDistance(lpnorms(k)._2), lpnorms(j)._2.fastDistance(lpnorms(k)._2)))
              graph(i)(j) = false
          }
        } 
      }
      
      /* Voting process */
      var votes = for(i <- 0 until lpnorms.length) yield HashMap[Double, Int]()
      for(i <- 0 until lpnorms.length){
        for(j <- 0 until lpnorms.length){
          if(graph(i)(j) && i != j) 
            votes(i) += lpnorms(j)._1 -> (votes(i).getOrElse(lpnorms(j)._1, 0) + 1)
        }
      }
      val preds = votes.map(_.maxBy(_._2)._1)
      
      if(secondOrder) {        
        /* 2nd order voting */
        //val svotes = votes.map(_.clone)
        for(i <- 0 until lpnorms.length if preds(i) != lpnorms(i)._1){
          for(j <- 0 until lpnorms.length){
            if(graph(i)(j) && i != j && lpnorms(i)._1 == lpnorms(j)._1){
              for(k <- 0 until lpnorms.length if graph(j)(k))
                votes(i) += lpnorms(j)._1 -> (votes(i).getOrElse(lpnorms(j)._1, 0) + 1)
            }
          }
        }
      }
      
      /* Decide whether to add the new example and to remove old noisy edges */
      val noisy = (0 until preds.length).map(i => preds(i) != lpnorms(i)._1)
      val toAdd = if(!noisy(0)) point else null // Accepted the insertion of new example
      val toRemove = neigs.zipWithIndex.filter(t => noisy(t._2 + 1)).map(_._1) // Remove noisy oldies
      (toAdd, toRemove)
      
    }    
    (edited.keys.filter(_ != null), edited.flatMap(_._2))
  }
}