package org.apache.spark.mllib.feature

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.knn.VectorWithNorm
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.feature.StreamingDistributedKNN._

/**
 * @author sramirez
 */



object InstanceSelection {
  
  case class TreeLPNorm(point: TreeLP, norm: VectorWithNorm, distances: Array[Float])
  
  private def computeDistances(neighbors: RDD[(TreeLP, Array[TreeLP])]) = {
    neighbors.map{ case (p, neigs) => 
      val elems = new TreeLPNorm(new TreeLP(p.point, p.itree, NOINSERT), new VectorWithNorm(p.point.features), Array.fill[Float](neigs.length + 1)(Float.PositiveInfinity)) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, NOREMOVE), new VectorWithNorm(q.point.features), Array.fill[Float](neigs.length + 1)(Float.PositiveInfinity)))
        
      /* Compute distances */
      for(i <- 0 until elems.length) { 
        val dist = new Array[Float](elems.length)
        for(j <- 0 until elems.length if i != j) {
          elems(i).distances(j) = elems(i).norm.fastDistance(elems(j).norm).toFloat
        }
      }  
      elems
    }    
  }
  
  def instanceSelection(elements: RDD[(TreeLP, Array[TreeLP])]) = {
    val processed = computeDistances(elements)
    val edited = processed.map(elems => localRNGE(elems))
    
    //println("Inserted in RNGE: " + edited.flatMap(x => x).filter(_.point.action == INSERT).count())    
    //println("Removed in RNGE: " + edited.flatMap(x => x).filter(_.point.action == REMOVE).count())
    
    edited.flatMap(x => x).filter(lp => lp.point.action == INSERT || lp.point.action == REMOVE)
      .map(_.point)
  }
  
  
  private def localRNGE(lpnorms: Array[TreeLPNorm], removeOldies: Boolean = false, secondOrder: Boolean = false) = {
        
      val graph = Array.fill[Boolean](lpnorms.length, lpnorms.length)(true)
      
      /* Compute the relative neighbor graph for edition (RNG-E) */
      for(i <- 0 until lpnorms.length) {
        for(j <- 0 until lpnorms.length if i != j) {
          for(k <- 0 until lpnorms.length if k != i && k != j) {
            if(lpnorms(i).distances(j) > math.max(lpnorms(i).distances(k), lpnorms(j).distances(k)))
              graph(i)(j) = false
          }
        } 
      }
      
      /* Voting process. We only check those elements related to the current insertion */
      val elementsToBeVoted = 0 +: (1 until lpnorms.length).filter(j => graph(j)(0))
      var votes = for(i <- 0 until lpnorms.length) yield HashMap[Double, Int]()
      for(i <- elementsToBeVoted){
        for(j <- 0 until lpnorms.length){
          val jlabel = lpnorms(j).point.point.label
          if(graph(i)(j) && i != j) 
            votes(i) += jlabel -> (votes(i).getOrElse(jlabel, 0) + 1)
        }
      }
      var preds = votes.map{ hm =>
        if(hm.isEmpty) None else Option(hm.maxBy(_._2)._1)
      }
      
      if(secondOrder) {        
        /* 2nd order voting */
        for(i <- elementsToBeVoted if preds(i).get != lpnorms(i).point.point.label){
          val ilabel = lpnorms(i).point.point.label
          for(j <- 0 until lpnorms.length){
            if(graph(i)(j) && i != j && ilabel == lpnorms(j).point.point.label){
              val jlabel = lpnorms(j).point.point.label
              for(k <- 0 until lpnorms.length if graph(j)(k))
                votes(i) += jlabel -> (votes(i).getOrElse(jlabel, 0) + 1)
            }
          }
        }
      }
      
      lpnorms.zipWithIndex.map{ case(lp, i) =>
        var action: Action = lp.point.action
        if(!votes(i).isEmpty) { // Elements to be voted
          val first = votes(i).head
          //val isEven = votes(i).filter{case (i, v) => v == first._2 }.size == votes(i).size
          val isEven = false
          if(!isEven || votes(i).size == 1) {
            val pred = votes(i).maxBy(_._2)._1
            if(i == 0 && lp.point.point.label == pred) {
              action = INSERT
            } else if (removeOldies && lp.point.point.label != pred && i > 0) {
              action = REMOVE
            }
          }
        }        
        new TreeLPNorm(new TreeLP(lp.point.point, lp.point.itree, action), lp.norm, lp.distances)
      }
  } 
}