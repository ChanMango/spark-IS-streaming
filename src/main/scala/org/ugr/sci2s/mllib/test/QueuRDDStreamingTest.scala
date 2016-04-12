package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StreamingDistributedKNN
import org.apache.spark.mllib.feature.StreamingDistributedKNN._

object QueuRDDStreamingTest {

  def main(args: Array[String]): Unit = {
    
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.

    val conf = new SparkConf().setMaster("local[4]").setAppName("MLStreamingTest")
    val ssc = new StreamingContext(conf, Seconds(5))
    val sc = ssc.sparkContext
    
    
    // Create a table of parameters (parsing)
    val params = args.map({arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  (param(0) -> param(1))
          case _ =>  ("" -> "")
        }
    }).toMap  
    
    println("Parameters: " + params.toString())
    
    //val input = params.getOrElse("input", "/home/sramirez/datasets/poker-5-fold/poker-5-1tra.data") 
    //val header = params.getOrElse("header", "/home/sramirez/datasets/poker-5-fold/poker.header") 
    
    val input = params.getOrElse("input", "/home/sramirez/datasets/kddcup_10_normal_versus_DOS-5-fold/kddcup_10_normal_versus_DOS-5-1tra.data") 
    val header = params.getOrElse("header", "/home/sramirez/datasets/kddcup_10_normal_versus_DOS-5-fold/kddcup_10_normal_versus_DOS.header") 
    val npart = params.getOrElse("npart", "4").toInt     
    val k = params.getOrElse("k", "1").toInt
    val rate = params.getOrElse("rate", "2500").toInt
    val seed = params.getOrElse("seed", "237597430").toLong
    val ntrees = params.getOrElse("ntrees", "4").toInt
    val overlap = params.getOrElse("overlap", "0.0").toDouble
    val edited = params.getOrElse("edited", "true").toBoolean
    val removeOld = params.getOrElse("removeOld", "false").toBoolean
    
    // Read file
    val typeConversion = KeelParser.parseHeaderFile(sc, header) 
    val bcTypeConv = sc.broadcast(typeConversion)
    val inputRDD = sc.textFile(input: String).map(line => KeelParser.parseLabeledPoint(bcTypeConv.value, line)).repartition(npart).cache()
    
    // Transform simple RDD into a QueuRDD for streaming
    //val inputRDD = sc.textFile(input).repartition().map(LabeledPoint.parse).cache
    val size = inputRDD.count()
    val nchunks = (size / rate).toInt
    val chunkPerc = 1.0 / nchunks
    println("Number of chunks: " + nchunks)
    println("Size of chunks: " + chunkPerc)    
    val arrayRDD = inputRDD.randomSplit(Array.fill[Double](nchunks)(chunkPerc), seed)
    val trainingData = ssc.queueStream(scala.collection.mutable.Queue(arrayRDD: _*), oneAtATime = true)
    
    val model = new StreamingDistributedKNN()
      .setNTrees(ntrees)
      .setOverlapDistance(overlap)
      .setEdited(edited)
      .setRemovedOld(removeOld)
      
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