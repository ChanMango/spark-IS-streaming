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
      val elems = new TreeLPNorm(new TreeLP(p.point, p.itree, NONE), new VectorWithNorm(p.point.features), Array.fill[Float](neigs.length + 1)(Float.PositiveInfinity)) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, NONE), new VectorWithNorm(q.point.features), Array.fill[Float](neigs.length + 1)(Float.PositiveInfinity)))
        
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
    val edited = processed.map(elems => localRNGE(elems)).cache
    
    println("Inserted in RNGE: " + edited.flatMap(x => x).filter(_.point.action == INSERT).count())    
    println("Removed in RNGE: " + edited.flatMap(x => x).filter(_.point.action == REMOVE).count())

    // Filter those groups where the main element is marked as not to be inserted
    // Then filter those neighbors marked to be removed
    val filtered = edited.filter(l => l(0).point.action != NONE).map{ neigs =>
      val nonRemoved = neigs.zipWithIndex.filter{ case (e, _) => e.point.action != REMOVE }
      nonRemoved.map{ case(flp, _) => 
        val ndist = nonRemoved.map{ case (_, j) => flp.distances(j) } 
        new TreeLPNorm(flp.point, flp.norm, ndist)
      }        
    }        
    
    val condensed = filtered.map(local1RNN).flatMap(y => y).filter(_.point.action != NONE).cache
    
    println("Inserted in RNN: " + condensed.filter(_.point.action == INSERT).count())    
    println("Removed in RNN: " + condensed.filter(_.point.action == REMOVE).count())
    
    edited.flatMap(y => y).filter(_.point.action == REMOVE).union(condensed).map(_.point)   
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
        var action: Action = if(lp.point.action != null) lp.point.action else NONE
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
  
  private def local1RNN(lpnorms: (Array[TreeLPNorm])) = {
    
      val S = Array.fill[Boolean](lpnorms.length)(true)
      if(lpnorms.length > 1 && lpnorms(0).point.action == INSERT) {
        val radius = lpnorms(0).distances.slice(1, lpnorms.length).max 
        val elementsToBeVoted = lpnorms(0).distances.zipWithIndex.filter{ case (d, _) => d < radius / 2 }.map(_._2)        
        val distances = lpnorms.map(_.distances.clone)
        
        val computeAcc = (i: Int) => {
          val j = distances(i).zipWithIndex.minBy(_._1)._2  
          if(lpnorms(i).point.point.label == lpnorms(j).point.point.label) 1 else 0
        }
        val iAcc = (0 until lpnorms.length).map(computeAcc).sum      
        
        for(i <- elementsToBeVoted) { // Only the instances inside the circle with radius = max / 2 are considered
          for(j <- 0 until lpnorms.length){ // Create a new set w/o this element
             distances(j)(i) = Float.PositiveInfinity
          }
          val nAcc = (0 until lpnorms.length).map(computeAcc).sum
          if(nAcc >= iAcc) { // Remove element, it's redundant
            S(i) = false
          } else {
            for(j <- 0 until lpnorms.length){ // Recover this element from the original matrix
              distances(j)(i) = lpnorms(i).distances(j)
            }
          }
        }        
      }
      
      /* Final result */
      lpnorms.zipWithIndex.map { case (lpn, i) => 
        var action: Action = if(lpn.point.action != null) lpn.point.action else NONE
        if (i > 0 && !S(i)) {
          action = REMOVE
        }
        new TreeLPNorm(new TreeLP(lpn.point.point, lpn.point.itree, action), lpn.norm, lpn.distances)        
      }
  } 
  
}