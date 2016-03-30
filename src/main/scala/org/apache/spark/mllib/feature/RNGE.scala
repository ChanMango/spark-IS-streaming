package org.apache.spark.mllib.feature

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{StreamingDistributedKNN => SDK}
import org.apache.spark.mllib.knn.VectorWithNorm
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import org.apache.spark.storage.StorageLevel

import org.apache.spark.mllib.feature.StreamingDistributedKNN._

/**
 * @author sramirez
 */



object RNGE {
  
  case class TreeLPNorm(point: TreeLP, norm: VectorWithNorm)
  
  def edition(neighbors: RDD[(TreeLP, Array[TreeLP])], secondOrder: Boolean = false) = {
    
    neighbors.flatMap{ case (p, neigs) => 
      
      val lpnorms = new TreeLPNorm(p, new VectorWithNorm(p.point.features)) +:
        neigs.map(q => new TreeLPNorm(q, new VectorWithNorm(q.point.features)))
        
      val graph = Array.fill[Boolean](lpnorms.length, lpnorms.length)(true)
      
      /* Compute the relative neighbor graph for edition (RNG-E) */
      for(i <- 0 until lpnorms.length) {
        for(j <- 0 until lpnorms.length) {
          val dij = lpnorms(i).norm.fastDistance(lpnorms(j).norm)
          for(k <- 0 until lpnorms.length if k != i && k != j) {
            if(dij > math.max(lpnorms(i).norm.fastDistance(lpnorms(k).norm), lpnorms(j).norm.fastDistance(lpnorms(k).norm)))
              graph(i)(j) = false
          }
        } 
      }
      
      /* Voting process */
      var votes = for(i <- 0 until lpnorms.length) yield HashMap[Double, Int]()
      for(i <- 0 until lpnorms.length){
        for(j <- 0 until lpnorms.length){
          val jlabel = lpnorms(j).point.point.label
          if(graph(i)(j) && i != j) 
            votes(i) += jlabel -> (votes(i).getOrElse(jlabel, 0) + 1)
        }
      }
      var preds = votes.map(_.maxBy(_._2)._1)
      
      if(secondOrder) {        
        /* 2nd order voting */
        for(i <- 0 until lpnorms.length if preds(i) != lpnorms(i).point.point.label){
          val ilabel = lpnorms(i).point.point.label
          for(j <- 0 until lpnorms.length){
            if(graph(i)(j) && i != j && ilabel == lpnorms(j).point.point.label){
              val jlabel = lpnorms(j).point.point.label
              for(k <- 0 until lpnorms.length if graph(j)(k))
                votes(i) += jlabel -> (votes(i).getOrElse(jlabel, 0) + 1)
            }
          }
        }
        preds = votes.map(_.maxBy(_._2)._1)
      }
      
      /* Decide whether to add the new example and to remove old noisy edges */
      val noisy = (0 until preds.length).map(i => preds(i) != lpnorms(i).point.point.label)
      p.action = INSERT // Accepted the insertion of new example (first)
      val toRemove = neigs.zipWithIndex.filter(t => noisy(t._2 + 1)).map{p => p._1.action = REMOVE; p._1} // Remove noisy oldies
      if(!noisy(0)) p +: toRemove else toRemove      
    }
  }
}