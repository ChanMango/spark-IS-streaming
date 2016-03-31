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
  
  case class TreeLPNorm(point: TreeLP, norm: VectorWithNorm)
  
  def localRNGE(neighbors: RDD[(TreeLP, Array[TreeLP])], secondOrder: Boolean = false) = {
    
    neighbors.flatMap{ case (p, neigs) => 
      
      val lpnorms = new TreeLPNorm(new TreeLP(p.point, p.itree, INSERT), new VectorWithNorm(p.point.features)) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, REMOVE), new VectorWithNorm(q.point.features)))
        
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
        for(i <- elementsToBeVoted if preds(i) != lpnorms(i).point.point.label){
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
      
      /* Decide whether to add the new example and to remove old noisy edges */
      lpnorms.zip(votes).filter{ case(lp, votes) => 
          if(votes.isEmpty){
            false
          } else {
            val first = votes.head
            val isEven = votes.filter{case (i, votes) => votes == first._2 }.size == votes.size
            if(!isEven) {
              val pred = votes.maxBy(_._2)._1
              if(lp.point.action == REMOVE) {
                lp.point.point.label != pred
              } else if (lp.point.action == INSERT) {
                lp.point.point.label == pred
              } else {
                false
              }
            } else {
              false
            }            
          }
        }.map(_._1.point)
    }
  } 
  
  def RNGE(neighbors: RDD[(TreeLP, Array[TreeLP])], secondOrder: Boolean = false) = {
    
    neighbors.flatMap{ case (p, neigs) => 
      
      val lpnorms = new TreeLPNorm(new TreeLP(p.point, p.itree, INSERT), new VectorWithNorm(p.point.features)) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, REMOVE), new VectorWithNorm(q.point.features)))
        
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
      lpnorms.zip(preds)
        .filter{ case(lp, pred) => (lp.point.point.label != pred && lp.point.action == REMOVE) || (lp.point.point.label == pred && lp.point.action == INSERT)}
        .map(_._1.point)
    }
  } 
  
  def CNN(neighbors: RDD[(TreeLP, Array[TreeLP])], k: Int, seed: Long) = {
    
    neighbors.flatMap{ case (p, neigs) => 
      
      val lpnorms = new TreeLPNorm(new TreeLP(p.point, p.itree, INSERT), new VectorWithNorm(p.point.features)) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, INSERT), new VectorWithNorm(q.point.features)))
        
      var S = new ArrayBuffer[Int]
      var result = new ArrayBuffer[TreeLP]
      val rand = new scala.util.Random(seed)
      
      /*Inserting one element per class*/
      val classes = lpnorms.map(_.point.point.label).distinct
      for(c <- classes) {
        var pos = rand.nextInt(lpnorms.length); var cont = 0
        while (lpnorms(pos).point.point.label != c && cont < lpnorms.length) {
			    pos = (pos + 1) % lpnorms.length
			    cont += 1
		    }        
        if (cont < lpnorms.length) {
			    S += pos
			    result += lpnorms(pos).point
		    }
      }

		  /*Algorithm body. We resort randomly the instances of T and compare with the rest of S.
		 	If an instance doesn't classified correctly, it is inserted in S*/
      var continue = false
  		do {
  		  continue = false
  		  val shuffled = scala.util.Random.shuffle(S.toSeq)
  		  S = S.sorted
  		  for(i <- 0 until lpnorms.length){  		    
  		    val pos = shuffled(i)
  		    val unselected = java.util.Arrays.binarySearch(S.asInstanceOf[Array[AnyRef]], pos) < 0
  		    if(unselected) {  		      
  		      val knn = S.map(idx => lpnorms(pos).norm.fastDistance(lpnorms(idx).norm) -> lpnorms(idx))
  		        .sortBy(-_._1).take(k)
  		      val pred = knn.map(_._2.point.point.label).groupBy(identity).maxBy(_._2.size)._1
  		      if(pred != lpnorms(pos).point.point.label){
  		        continue = true
  		        S += pos
  		        result += lpnorms(pos).point
  		        S = S.sorted
  		      }
  		    }		      
  		  }
  		} while (continue)
  		result
    }
  } 
  
  
}