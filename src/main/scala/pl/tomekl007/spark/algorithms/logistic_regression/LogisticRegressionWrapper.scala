package pl.tomekl007.spark.algorithms.logistic_regression

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import pl.tomekl007.spark.algorithms.Word2VecImpl
import pl.tomekl007.spark.{CsvUtil, RunConfig}

import scala.collection.mutable

object LogisticRegressionWrapper {
  val NumClasses = 2


  def apply(dataFrame: DataFrame, sc: SparkContext, runConfig: RunConfig) = {

    val result: mutable.Map[String, Statistic] = mutable.Map()
    val authors = dataFrame.rdd.map(row => row.getAs[String]("author")).distinct().collect()

    val models = authors.map(author => countForAuthor(dataFrame, author, result, runConfig))
    println("sort by precision:")
    result.toSeq.sortBy(_._2.precision).reverse.foreach(println)
    println("sort by area under roc:")
    result.toSeq.sortBy(_._2.areaUnderROC).reverse.foreach(println)
    CsvUtil.writeAsCsv(result.toSeq.sortBy(_._2.areaUnderROC).reverse.toMap,
      s"vect_size=${runConfig.W2VVectSize}_min=${runConfig.W2VMinCount}_numClasses=${runConfig.LRNumClasses}")

    evaluateModel(models)
  }


  def applyOneModel(dataFrame: DataFrame, sc: SparkContext, runConfig: RunConfig) = {

    val result: mutable.Map[String, Statistic] = mutable.Map()
    val model = Array(oneModelForAllAuthors(dataFrame, result, runConfig))
    CsvUtil.writeAsCsv(result.toSeq.sortBy(_._2.areaUnderROC).reverse.toMap,
      s"vect_size=${runConfig.W2VVectSize}_min=${runConfig.W2VMinCount}_numClasses=${runConfig.LRNumClasses}")

    //evaluateModel(model)
  }

  def evaluateModel(models: Array[(String, LogisticRegressionModel, Double, GaussianMixtureModel)]): Unit = {

    while (true) {
      try {
        //format: This is a text/23
        val userInput = readLine
        println("input:" + userInput)
        val input = createUserInput(userInput)
        val wordVector = Word2VecImpl(input._1, RunConfig())
        println("wordVect " + wordVector)
        models.map { case (author, model, areaUnderRoc, gmm) =>
          model.clearThreshold()
          val gmmRes = gmm.predictSoft(Vectors.dense(input._2)).toList
          (author, model.predict(wordVector), areaUnderRoc, gmmRes, gmmRes.sum)
        }.sortBy(_._2).reverse.foreach(println)
      } catch {
        case e: Exception => print(e); print(e.getStackTrace.toList)
        case e: RuntimeException => print(e); print(e.getStackTrace.toList)
      }
    }
  }

  def createUserInput(userInput: String): (String, Int) = {
    val i = userInput.split("/")
    val text = i(0)
    val hour = i(1).trim.toInt
    (text, hour)
  }

  def oneModelForAllAuthors(dataFrame: DataFrame, result: mutable.Map[String, Statistic],
                            runConfig: RunConfig) = {

    val data = dataFrame.rdd.repartition(8)
      .map { row =>
        val author = row.getAs[String]("author")
        val label = if (author.equals("wÅ‚adekv")) 1.0
        else if (author.equals("Dzik")) 2.0
        else if (author.equals("Szalony")) 3.0
        else if (author.equals("Szczepan")) 4.0
        else 0.0 //each author has own model - n to many

        val vector = row.getAs[Vector]("result")
        LabeledPoint(label, vector) //label - first author = 1.0, second = 2.0,..
      }.cache()


    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val modelSetup1 = new LogisticRegressionWithLBFGS()
      .setNumClasses(runConfig.LRNumClasses) //one versus one, one versus many

    //    modelSetup.optimizer.setRegParam(1.0)
    val model1 = modelSetup1.run(training)

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model1.predict(features)
      (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val metrics2 = new BinaryClassificationMetrics(predictionAndLabels)
    printMetrics(metrics, result, "all", data.count(), metrics2.areaUnderROC(), model1.getThreshold.get)
    ("all", model1, metrics2.areaUnderROC())
  }

  def countForAuthor(dataFrame: DataFrame, author: String, result: mutable.Map[String, Statistic],
                     runConfig: RunConfig): (String, LogisticRegressionModel, Double, GaussianMixtureModel) = {

    println(s"LogisticRegression for author: $author")

    val gmm = GMMWrapper(dataFrame, author)

    val data = dataFrame.rdd.repartition(8)
      .map { row =>
        val label = if (row.getAs[String]("author").equals(author)) 1.0 else 0.0
        val vector = row.getAs[Vector]("result")
        LabeledPoint(label, vector)
      }.cache()

    val numberOfPosts = data.filter(lp => lp.label == 1.0).count()

    // Split data into training (60%) and test (40%). //cross validation - validate on different, training on different
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val modelSetup = new LogisticRegressionWithLBFGS()
      .setNumClasses(runConfig.LRNumClasses)

    modelSetup.optimizer.setRegParam(0.0) //todo

    val model = modelSetup.run(training)

    //todo http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html

    //todo to see if model is not too trained run verification against training set
    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    //todo make cross-validation http://spark.apache.org/docs/latest/ml-guide.html#example-model-selection-via-cross-validation
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val metrics2 = new BinaryClassificationMetrics(predictionAndLabels)
    //todo what is are under roc http://gim.unmc.edu/dxtests/roc3.htm
    //todo area under pr http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    printMetrics(metrics, result, author, numberOfPosts, metrics2.areaUnderROC(), model.getThreshold.get)
    (author, model, metrics2.areaUnderROC(), gmm)

    // Save and load model
    //    model.save(sc, "myModelPathForAuthor-dr_know")
    //    val sameModel = LogisticRegressionModel.load(sc, "myModelPathForAuthor-dr_know")
  }

  def printMetrics(metrics: MulticlassMetrics, resultMap: scala.collection.mutable.Map[String, Statistic],
                   author: String, numberOfPosts: Long, areaUnderROC: Double, threshold: Double): Unit = {
    println("recall = " + metrics.recall)
    println("Precision = " + metrics.precision)
    println("FalsePositive = " + metrics.weightedFalsePositiveRate)
    println("TruePositive = " + metrics.weightedTruePositiveRate)
    println("areaUnderROC = " + areaUnderROC)
    resultMap.put(author, Statistic(author, metrics.precision, metrics.recall,
      metrics.weightedFalsePositiveRate, metrics.weightedTruePositiveRate, numberOfPosts, areaUnderROC, threshold))
    //todo why precision and recall is the same ?

  }

  //todo (Dzik,Statistic(Dzik,0.9112049689440994,0.9112049689440994,0.5049597164027653,0.9112049689440993,5238,0.703122626270667,0.5))
  //falsePositiveRate - 0.5049597164027653 -- should be low
  //truePositiveRate - 0.9112049689440993 -- should be high
  //area under roc - 0.703122626270667 ->
  //  .90-1 = excellent (A)
  //  .80-.90 = good (B)
  //  .70-.80 = fair (C)
  //  .60-.70 = poor (D)
  //  .50-.60 = fail (F)
  //https://en.wikipedia.org/wiki/Precision_and_recall
  //precision - 0.9112049689440994 -- how many selected items are relevant ?
  //recall - 0.9112049689440994 -- how many relevant items are selected ?
  case class Statistic(author: String, precision: Double, recall: Double, falsePositiveRate: Double, truePositiveRate: Double,
                       numberOfPosts: Long, areaUnderROC: Double, threshold: Double)

}
