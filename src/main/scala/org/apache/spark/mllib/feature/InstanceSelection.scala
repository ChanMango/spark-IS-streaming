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

    val filtered = edited.map{ neigs =>
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
  
  private def localRNGE(lpnorms: Array[TreeLPNorm], secondOrder: Boolean = false) = {
        
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
        if(!votes(i).isEmpty) {
          val first = votes(i).head
          val isEven = votes(i).filter{case (i, v) => v == first._2 }.size == votes(i).size
          if(!isEven || votes(i).size == 1) {
            val pred = votes(i).maxBy(_._2)._1
            if(i == 0 && lp.point.point.label == pred) {
              action = INSERT
            } else if (i > 0 && lp.point.point.label != pred) {
              action = REMOVE
            }
          }
        }        
        new TreeLPNorm(new TreeLP(lp.point.point, lp.point.itree, action), lp.norm, lp.distances)
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
  
  private def local1RNN(lpnorms: (Array[TreeLPNorm])) = {
    
      val radius = lpnorms(0).distances.slice(1, lpnorms.length).max
      val elementsToBeVoted = 0 +: lpnorms(0).distances.zipWithIndex.filter{ case (d, _) => d < radius / 2 }.map(_._2)
      val S = Array.fill[Boolean](lpnorms.length)(true)
      val distances = lpnorms.map(_.distances.clone)
      
      val computeAcc = (i: Int) => {
        val j = distances(i).zipWithIndex.minBy(_._1)._2  
        if(lpnorms(i).point.point.label == lpnorms(j).point.point.label) 1 else 0
      }
      val iAcc = (0 until lpnorms.length).map(computeAcc).sum      
      
      for(i <- elementsToBeVoted) { // Only the instances inside the circle with radius = max /2 are considered
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
      
      /* Voting process */
      /*for(i <- elementsToBeVoted) {
        val ilabel = lpnorms(i).point.point.label
        var j = 0; var finish = false
        var loses = 0; var wins = 0
        while(j < lpnorms.length && !finish) {
          val inddist = lpnorms(j).distances.zipWithIndex
          val closest = inddist.minBy(_._1)
          if(closest._1 < Float.PositiveInfinity) {
            if(closest._2 == i) {
              inddist(closest._2) = Float.PositiveInfinity -> closest._2 
              // Find the new 1-NN for this element
              val nclosest = inddist.minBy(_._1)
              if(nclosest._1 < Float.PositiveInfinity) {
                if(lpnorms(closest._2).point.point.label == ilabel && lpnorms(nclosest._2).point.point.label != ilabel) { 
                  // Misclassified
                  loses += 1
                } else if (lpnorms(closest._2).point.point.label != ilabel && lpnorms(nclosest._2).point.point.label == ilabel) {
                  // Correctly classified
                  wins += 1
                }
              } else {
                finish = true
              }
            }          
          } else {
            finish = true
          }          
          j += 1
        }
        // Update references to the instance already removed
        if(wins < loses) {
          S(i) = false
          for (z <- 0 until lpnorms.length) lpnorms(z).distances(i) = Float.PositiveInfinity 
        }
      }*/
      
      /* Final result */
      lpnorms.zipWithIndex.map{ case (lpn, i) => 
        var action: Action = if(lpn.point.action != null) lpn.point.action else NONE
        if(i == 0 && S(i)) {
          action = INSERT
        } else if (i > 0 && !S(i)) {
          action = REMOVE
        }
        new TreeLPNorm(new TreeLP(lpn.point.point, lpn.point.itree, action), lpn.norm, lpn.distances)        
      }
  } 
  
}