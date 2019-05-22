package pl.tomekl007.spark.algorithms

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/**
  * Created by tomasz.lelek on 16/03/16.
  */
//https://blogs.msdn.microsoft.com/bigdatasupport/2015/09/24/a-kmeans-example-for-spark-mllib-on-hdinsight/
object KMeansClustering {
  def apply(input: DataFrame): Unit = {

    // Load and parse the data
    val vect = input.rdd
      .map(row => row.getAs[org.apache.spark.mllib.linalg.Vector]("result"))
//    val parsedData = vect.map(s => Vectors.dense(s.toArray))

    // Cluster the data into two classes using KMeans
    val numClusters = 100
    val numIterations = 20
    val clusters = KMeans.train(vect, numClusters, numIterations)


    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(vect)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
//    clusters.save(sc, "myModelPath")
//    val sameModel = KMeansModel.load(sc, "myModelPath")

  }
}
