package pl.tomekl007.spark.algorithms.logistic_regression

import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.SQLContext
import org.joda.time.{DateTimeZone, DateTime}
import org.scalatest.FunSuite
import org.scalatest.Matchers._
import pl.tomekl007.spark.SparkContextInitializer

class GMMWrapperTest extends FunSuite {
  test("Should predict") {
    //given
    val spark = SparkContextInitializer.createSparkContext("test", List.empty)
    val data = SQLContext.getOrCreate(spark).createDataFrame(List(TestObj("a", DateTime.now().getMillis),
      TestObj("a", DateTime.now().plusHours(12).getMillis), TestObj("a", DateTime.now().plusHours(6).getMillis)))

    //when
    val gmm = GMMWrapper(data, "a")

    //then
    val vect = Vectors.dense(DateTime.now().getMillis)
    val res = gmm.predict(vect)
    println(res)
    res shouldEqual 1
  }


  test("should normalize time") {
    //given
    val timestamp = 1341587905000L

    //when
    val res = GMMWrapper.normalizeTime("ignore", timestamp)
    val resDate = new DateTime(res.toLong, DateTimeZone.UTC)

    //then
    resDate.getYear.shouldEqual(2012)
    resDate.getDayOfMonth.shouldEqual(6)
    resDate.getMonthOfYear.shouldEqual(7)
    resDate.getHourOfDay.shouldEqual(15)
    resDate.getMinuteOfHour.shouldEqual(18)
    resDate.getSecondOfMinute.shouldEqual(0)
    resDate.getMillisOfSecond.shouldEqual(0)


  }


  case class TestObj(author: String, createdTimestamp: Long)

}
