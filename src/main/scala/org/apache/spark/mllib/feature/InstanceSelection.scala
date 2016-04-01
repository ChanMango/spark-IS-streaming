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
  
  case class TreeLPNorm(point: TreeLP, norm: VectorWithNorm, var distances: Array[Float])
  
  private def computeDistances(neighbors: RDD[(TreeLP, Array[TreeLP])]) = {
    neighbors.map{ case (p, neigs) => 
      val elems = new TreeLPNorm(new TreeLP(p.point, p.itree, INSERT), new VectorWithNorm(p.point.features), null) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, REMOVE), new VectorWithNorm(q.point.features), null))
      //val distances = Array.fill[Double](elems.length, elems.length)(Double.PositiveInfinity)
      
      /* Compute distances */
      for(i <- 0 until elems.length) {
        val dist = new Array[Float](elems.length)
        for(j <- 0 until elems.length if i != j) {
          dist(j) = elems(i).norm.fastDistance(elems(j).norm).toFloat
        } 
        elems(i).distances = dist
      }  
      elems
    }    
  }
  
  def instanceSelection(elements: RDD[(TreeLP, Array[TreeLP])]) = {
    val processed = computeDistances(elements)
    val edited = localRNGE(processed).cache
    
    val filtered = edited.map{ neigs =>
      val nonRemoved = neigs.zipWithIndex.filter{ case (e, i) => e.point.action != REMOVE }
      nonRemoved.map{ case(flp, _) => 
        val ndist = nonRemoved.map{ case (_, j) => flp.distances(j) } 
        flp.distances = ndist
        flp
      }        
    }    
    val condensed = local1RNN(filtered).flatMap(y => y).filter(_.point.action != NONE)
    edited.flatMap(y => y).filter(_.point.action == REMOVE).union(condensed).map(_.point)   
  }
  
  private def localRNGE(neighbors: RDD[Array[TreeLPNorm]], secondOrder: Boolean = false) = {
    
    neighbors.map{ lpnorms => 
        
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
        val first = votes(i).head
        val isEven = votes(i).filter{case (i, v) => v == first._2 }.size == votes(i).size
        if(!isEven) {
          val pred = votes(i).maxBy(_._2)._1
          if(i == 0 && lp.point.point.label == pred) {
            lp.point.action = INSERT
          } else if (i > 0 && lp.point.point.label != pred) {
            lp.point.action = REMOVE
          } else {
            lp.point.action = NONE
          }
        } else {
          lp.point.action = NONE
        }
        lp
      }
    }
  } 
  
  private def CNN(neighbors: RDD[(TreeLP, Array[TreeLP])], k: Int, seed: Long) = {
    
    neighbors.flatMap{ case (p, neigs) => 
      
      val lpnorms = new TreeLPNorm(new TreeLP(p.point, p.itree, INSERT), new VectorWithNorm(p.point.features), null) +:
        neigs.map(q => new TreeLPNorm(new TreeLP(q.point, q.itree, INSERT), new VectorWithNorm(q.point.features), null))
        
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
  
  private def local1RNN(neighbors: RDD[(Array[TreeLPNorm])]) = {
    
    neighbors.map{ lpnorms =>       
      
      val radius = lpnorms(0).distances.max
      val elementsToBeVoted = lpnorms(0).distances.zipWithIndex.filter{ case (d, _) => d < radius / 2 }.map(_._2)
      val redundant = Array.fill[Boolean](lpnorms.length)(false)
      
      /* Voting process */
      for(i <- elementsToBeVoted) {
        val ilabel = lpnorms(i).point.point.label
        redundant(i) = true
        var j = 0; var finish = false
        while(j < lpnorms.length && !finish) {
          val inddist = lpnorms(j).distances.zipWithIndex
          val closest = inddist.minBy(_._1)
          if(closest._1 < Float.PositiveInfinity) {
              if(closest._2 == i) {
                // i was its 1-NN
                if(lpnorms(closest._2).point.point.label != ilabel) {
                // but it is classified incorrectly
                redundant(i) = false
                finish = true       
              } else {
                inddist(closest._2) = Float.PositiveInfinity -> closest._2 
                // Find the new 1-NN for this element
                val nclosest = inddist.minBy(_._1)
                if(nclosest._1 < Float.PositiveInfinity) {
                  if(lpnorms(nclosest._2).point.point.label != ilabel) { 
                    // If i is incorrectly classified by its new 1-NN
                    redundant(i) = false
                    finish = true
                  }
                } else {
                  finish = true
                }
              }
            }            
          } else {
            finish = true
          }          
          j += 1
        }
        // Update references to the instance already removed
        if(redundant(i)) {
          for (z <- 0 until lpnorms.length) lpnorms(z).distances(i) = Float.PositiveInfinity 
        }
      }
      
      /* Final result */
      lpnorms.zipWithIndex.map{ case (lpn, i) => 
        if(i == 0 && !redundant(i)) {
          lpn.point.action = INSERT
        } else if (i > 0 && redundant(i)) {
          lpn.point.action = REMOVE
        } else {
          lpn.point.action = NONE
        }
        lpn        
      }
    }
  } 
  
}