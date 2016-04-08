package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature._
import org.apache.spark.annotation.Since

object QueuRDDStreamingTest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val conf = new SparkConf().setMaster("local[4]").setAppName("MLStreamingTest")
    val ssc = new StreamingContext(conf, Seconds(2))
    val sc = ssc.sparkContext
    
    
    // Create a table of parameters (parsing)
    val params = args.map({arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  (param(0) -> param(1))
          case _ =>  ("" -> "")
        }
    }).toMap  
    
    val input = params.getOrElse("input", "tmp")    
    val k = params.getOrElse("k", "1").toInt
    val rate = params.getOrElse("rate", "5000").toInt
    val seed = params.getOrElse("seed", "237597430L").toLong
    
    val inputRDD = sc.textFile(input).map(LabeledPoint.parse).cache
    val size = inputRDD.count()
    val chunkSize = rate.toFloat / size
    val nchunks = math.ceil(size * chunkSize).toInt
    println("Number of chunks: " + nchunks)
    println("Size of chunks: " + chunkSize)
    
    val arrayRDD = inputRDD.randomSplit(Array.fill[Double](nchunks)(chunkSize), seed)
    val trainingData = ssc.queueStream(scala.collection.mutable.Queue(arrayRDD: _*), oneAtATime = true)
    
    val model = new StreamingDistributedKNN().setNPartitions(4)
    val preds = model.predictOnValues(trainingData, k)
        .map{case (label, pred) => if(label == pred) (1, 1)  else (0, 1)}
        .reduce{ case (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2) }
        .map{ case (wins, sum) => wins / sum.toFloat}
        .print
    model.trainOn(trainingData)
    

    ssc.start()
    ssc.awaitTermination()
  }

}