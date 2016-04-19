package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StreamingDistributedKNN
import org.apache.spark.mllib.feature.StreamingDistributedKNN._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.scheduler.StreamingListener
import org.apache.spark.streaming.scheduler.StreamingListenerBatchCompleted
import org.apache.spark.streaming.scheduler.StreamingListenerBatchStarted
import org.apache.spark.streaming.scheduler.StreamingListenerBatchCompleted
import mtree.DataLP
import mtree.IndexedLP
import org.apache.spark.mllib.knn.VectorWithNorm
import mtree.MTree
import org.apache.spark.mllib.linalg.Vectors
import java.util.Calendar

object QueuRDDStreamingTest extends Logging {

  def main(args: Array[String]): Unit = {   
    
    // Create a table of parameters (parsing)
    val params = args.map({arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  (param(0) -> param(1))
          case _ =>  ("" -> "")
        }
    }).toMap  
    
    println("Parameters: " + params.toString())
    
    val fileType = params.getOrElse("type", "keel") 
    val input = params.getOrElse("input", "/home/sramirez/datasets/poker-5-fold/poker-5-1tst.data") 
    val header = params.getOrElse("header", "/home/sramirez/datasets/poker-5-fold/poker.header") 
    val output = params.getOrElse("output", "/home/sramirez/datasets/poker-5-fold/output") 
    val npart = params.getOrElse("npart", "4").toInt     
    val k = params.getOrElse("k", "1").toInt
    val rate = params.getOrElse("rate", "2500").toInt
    val interval = params.getOrElse("interval", "1000").toLong
    val seed = params.getOrElse("seed", "237597430").toLong
    val ntrees = params.getOrElse("ntrees", "4").toInt
    val overlap = params.getOrElse("overlap", "0.0").toDouble
    val edited = params.getOrElse("edited", "false").toBoolean
    val removeOld = params.getOrElse("removeOld", "false").toBoolean
    val timeout = params.getOrElse("timeout", "600000").toLong
    val kGraph = params.getOrElse("kgraph", "10").toInt
        
    // Create a local StreamingContext with two working thread and batch interval of 1 second.
    // The master requires 2 cores to prevent from a starvation scenario.
    val conf = new SparkConf().setAppName("MLStreamingTest")
      //.setMaster("local[4]")
    conf.registerKryoClasses(Array(classOf[DataLP], classOf[IndexedLP], 
        classOf[TreeLP], classOf[VectorWithNorm]))
    val ssc = new StreamingContext(conf, Milliseconds(interval))
    val sc = ssc.sparkContext
    
    // Read file    
    val inputRDD: RDD[LabeledPoint] = if(fileType == "keel") {
      val typeConversion = KeelParser.parseHeaderFile(sc, header) 
      val bcTypeConv = sc.broadcast(typeConversion)
      sc.textFile(input: String).repartition(npart).map(line => KeelParser.parseLabeledPoint(bcTypeConv.value, line))
    } else if(fileType == "labeled") {      
      sc.textFile(input: String).repartition(npart).map(LabeledPoint.parse)
    } else {
      sc.textFile(input: String).filter(line => !line.startsWith("#")).repartition(npart).map{ line =>         
          val arr = line split ","         
          val label = arr.head.toDouble
          val features = arr.slice(1, arr.length).map(_.toDouble)
          new LabeledPoint(label, Vectors.dense(features))
      }
    } 
    
    // Transform simple RDD into a QueuRDD for streaming
    inputRDD.cache()
    val size = inputRDD.count()
    logInfo("Distinct: " + inputRDD.map(_.features).distinct().count())
    val nchunks = (size / rate).toInt
    val chunkPerc = 1.0 / nchunks
    logInfo("Number of batches: " + nchunks)
    logInfo("Batch size: " + chunkPerc)    
    val arrayRDD = inputRDD.randomSplit(Array.fill[Double](nchunks)(chunkPerc), seed).map(_.cache())
    logInfo("Count by partition: " + arrayRDD.map(_.count()).mkString(",")) // Important to force the persistence
    inputRDD.unpersist()
    val trainingData = ssc.queueStream(scala.collection.mutable.Queue(arrayRDD: _*), 
        oneAtATime = true)
        
    val listen = new MyJobListener(ssc, nchunks)
    ssc.addStreamingListener(listen)
        
    val model = new StreamingDistributedKNN()
      .setNTrees(ntrees)
      .setOverlapDistance(overlap)
      .setEdited(edited)
      .setRemovedOld(removeOld) 
      .setSeed(seed)
      .setKGraph(kGraph)
    
    val preds = model.predictOnValues(trainingData, k)
        .map{case (label, pred) => if(label == pred) (1, 1)  else (0, 1)}
        .reduce{ case (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2) }
        .map{ case (wins, sum) => Calendar.getInstance().getTime() + " - Accuracy per batch: " + (wins / sum.toFloat)}
        .print()
    model.trainOn(trainingData)
    
    ssc.start()
    ssc.awaitTerminationOrTimeout(timeout)
  }
  
  private class MyJobListener(ssc: StreamingContext, totalBatches: Long) extends StreamingListener {
    private var nBatches: Long = 0
    override def onBatchCompleted(batchCompleted: StreamingListenerBatchCompleted) {   
      if(nBatches < totalBatches) 
        nBatches += 1
      logInfo("Number of batches processed: " + nBatches)
      val binfo = batchCompleted.batchInfo
      logInfo("Batch scheduling delay: " + binfo.schedulingDelay.getOrElse(0L))
      logInfo("Batch processing time: " + binfo.processingDelay.getOrElse(0L))
    }
  }

}