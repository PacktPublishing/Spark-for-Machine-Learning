package pl.tomekl007.spark.algorithms.logistic_regression

import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.joda.time.{DateTime, DateTimeZone}

/**
  * Created by tomasz.lelek on 28/04/16.
  */
object GMMWrapper {
  def apply(dataFrame: DataFrame, author: String): GaussianMixtureModel = {
    // Load and parse the data
    val timeOfPostsForAuthor = dataFrame.rdd
      .map(row => (row.getAs[String]("author"), row.getAs[Long]("createdTimestamp")))
      .map(r => (r._1, normalizeTime(r)))
      .filter(a => a._1.equalsIgnoreCase(author))
      .groupByKey()

    timeOfPostsForAuthor.foreach(println)

    val parsedData = timeOfPostsForAuthor
      .flatMap(_._2)
      .map(elem => Vectors.dense(elem))
      .cache()

    // Cluster the data into two classes using GaussianMixture
    val gmm = new GaussianMixture()
      .setK(3) //experiment with number of gaussian
      .run(parsedData)
    // output parameters of max-likelihood model
    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }
    gmm
  }

  def normalizeTime(r: (String, Long)): Double =
    new DateTime(r._2, DateTimeZone.UTC).getHourOfDay
}
