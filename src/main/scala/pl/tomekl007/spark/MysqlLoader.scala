package pl.tomekl007.spark

import org.apache.spark.sql.{DataFrame, SQLContext}
import pl.tomekl007.spark.algorithms.logistic_regression.LogisticRegressionWrapper
import pl.tomekl007.spark.algorithms.{Word2VecImpl, Word2VectWithModel}
import pl.tomekl007.spark.model.Message
import pl.tomekl007.spark.utils.InputCleanUtil

import scala.collection.mutable.ArrayBuffer


//mysql.server start
//mysql -u root
//Automatyczne dzielenie użtkowników serwisu internetowego na kategorie
//jak zrobic vect ( bow ) ktory reprezentuje cechy dystynkywne
//regresja liniowa na bow authora
//gmm na time posta
//naniesc to na siebie
class MysqlLoader(sqlContext: SQLContext, runConfig: RunConfig) {

  def load(): Unit = {

    val dataframe_mysql =
      sqlContext.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost/forum")
        .option("driver", "com.mysql.jdbc.Driver")
        .option("dbtable", "b_messages")
        .option("user", "root")
        .option("password", "").load()

    //    val slicedRdd = cleanedRdd.map(strings => MysqlLoader.slice(strings.toArray, 3))
    //    slicedRdd.foreach(_.toList.foreach(l => println(l.toList)))

    //        val slicedRdd = cleanedRdd.map(strings => Array(strings.toArray))
    //            PrefixSpan(slicedRdd)
    //            LDA(cleanedRdd)


    //    SemanticAnalysis(sqlContext.sparkContext.parallelize(List("czesc co u Ciebie slychac Tomek ?", "Dziękuje że pytasz - odparł Tomek")), StopWords.allStopWords.toSet)
    //    SemanticAnalysis(dataframe_mysql.rdd.map(row => row.getAs[String]("body")), StopWords.polishStopWords.toSet)
    //Proszę podejść do tego inaczej, model (sieć czy co tam Pan sobie wybierze) może rozpoznawać najczęściej piszących użytkowników, a resztę klasyfikować jako "inni". Można wtedy przebadać np. skuteczność rozpoznawania w zależności od ilości postów. Przykładowo zrobić próg odcięcia na poziomie 100, 500 i 1000 postów (poniżej tej ilości wpadają do "inni").
    //Zresztą proszę też wziąć pod uwagę że ma Pan też inne "cenne" dane np. datę i godzinę powstania posta która jest dosyć charakterystyczna dla części użytkowników.


    val cleanedDf = InputCleanUtil.tokenizeAndStopWords(dataframe_mysql, sqlContext)
    val topAuthors = getTopAuthorsFromNumberOfPosts(cleanedDf, 3000) //todo show how this is grouped when 100, 500, 1000 posts' top authors
    val vectorized = Word2VecImpl(topAuthors, runConfig)

//    LogisticRegressionWrapper.applyOneModel(vectorized.dataFrame, sqlContext.sparkContext, runConfig)
    LogisticRegressionWrapper(vectorized.dataFrame, sqlContext.sparkContext, runConfig)



  }


  def findSynonyms(vectorized: Word2VectWithModel): Unit = {
    while (true) {
      val userinput = readLine
      try {
        val res = vectorized.model.findSynonyms(userinput, 5)
        res.show(5)
      } catch {
        case e: IllegalStateException => println(e)
      }

    }
  }

  def getTopAuthorsFromNumberOfPosts(df: DataFrame, numberOfPosts: Int): DataFrame = {
    import sqlContext.implicits._
    df.rdd.map(row => (row.getAs[String]("author"),
      row.getAs[Seq[String]]("words"), row.getAs[Long]("createdTimestamp"), row.getAs[Long]("messageId"), row.getAs[String]("subject")))
      .groupBy(_._1)
      .map(x => (x._1, x._2, x._2.size))
      .sortBy(_._3, ascending = false)
      .filter(x => x._3 > numberOfPosts)
      .map(_._2.map(x => Message(x._2, x._1, x._3, x._4, x._5)))
      .flatMap(identity)
      .filter(_.words.nonEmpty)
      .toDF()
  }


  def morfologicAnalysis(words: DataFrame): Unit = {
    //    System.setProperty( "java.library.path", "/Users/tomasz.lelek/Downloads/morfeusz" )
    //    val morfeusz = Morfeusz.createInstance()
    //    words
    //      .rdd
    //      .map(row => row.getAs[String]("words"))
    //      .map(w => morfeusz.analyseAsList(w))
    //      .foreach(println)

    //todo maybe lucene https://lucene.apache.org/core/4_2_1/analyzers-stempel/overview-summary.html

  }

}


object MysqlLoader {
  def main(args: Array[String]): Unit = {
    val sparkContext = SparkContextInitializer.createSparkContext("mysqlLoader", List.empty)
    val runConfig = RunConfig(2, 5, 200)
    new MysqlLoader(SQLContext.getOrCreate(sparkContext), runConfig).load()
  }

  def slice(input: Array[String], windowSize: Int): Array[Array[String]] = {
    loop(input, windowSize, ArrayBuffer.empty, 0)
  }

  def loop(input: Array[String], windowSize: Int, res: ArrayBuffer[Array[String]], index: Int): Array[Array[String]] = {
    if (windowSize + index < input.length) {
      res += input.slice(index * windowSize, index + 1 * windowSize)
      loop(input, windowSize, res, index + 1)
    } else {
      res.toArray
    }
  }


}