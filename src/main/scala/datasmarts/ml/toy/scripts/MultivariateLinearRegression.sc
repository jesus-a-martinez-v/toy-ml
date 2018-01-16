import com.github.tototoshi.csv._
import scala.util.Random

sealed trait Data

case class Numeric(value: Double) extends Data
case class Text(value: String) extends Data

def loadCsv(filePath: String): Vector[Vector[Data]] = {
  val reader = CSVReader.open(filePath)

  reader
    .toStream
    .map(x => x.toArray.map(Text).toVector)
    .toVector
}

def textColumnToNumeric(data: Vector[Vector[Data]], columnIndex: Int) = {
  data.map { row =>
    val (firstHalf, secondHalf) = row.splitAt(columnIndex)
    val affectedValue =
      secondHalf.head match {
        case Text(value) => Numeric(value.toDouble)
        case d => d
      }

    firstHalf ++ Vector(affectedValue) ++ secondHalf.tail
  }
}

def categoricalColumnToNumeric(data: Vector[Vector[Data]], columnIndex: Int) = {
  val uniqueColumnValues = data.foldLeft(Set[Data]()) { (set, row) =>
    set + row(columnIndex)
  }

  val lookUpTable = uniqueColumnValues.zipWithIndex.toMap

  val categorizedData = data.map { row =>
    val (firstHalf, secondHalf) = row.splitAt(columnIndex)
    val affectedValue = Numeric(lookUpTable(secondHalf.head).toDouble)

    firstHalf ++ Vector(affectedValue) ++ secondHalf.tail
  }

  (categorizedData, lookUpTable)
}

type Dataset = Vector[Vector[Data]]
type MinMaxData = Vector[Option[(Double, Double)]]
type StatisticData = Vector[Option[Double]]

def isNumeric(data: Data) = data match {
  case _: Numeric => true
  case _ => false
}

def isText(data: Data) = !isNumeric(data)


def getNumericValue(data: Data): Option[Double] = data match {
  case Numeric(value) => Some(value)
  case _ => None
}

def getTextValue(data: Data): Option[String] = data match {
  case Text(value) => Some(value)
  case _ => None
}

def getDatasetMinAndMax(dataset: Dataset): MinMaxData = {
  if (dataset.isEmpty) {
    Vector.empty
  } else {
    val numberOfColumns = dataset.head.length
    val columnIndicesRange = (0 until numberOfColumns).toVector
    val testRow = dataset.head

    for {
      columnIndex <- columnIndicesRange
    } yield {
      if (isText(testRow(columnIndex))) {
        None
      } else {
        val columnValues = dataset.map { row =>
          getNumericValue(row(columnIndex)).get
        }.sorted

        val max = columnValues.last
        val min = columnValues.head

        Some((min, max))
      }
    }
  }
}

def normalizeDataset(dataset: Dataset, minMaxes: MinMaxData): Dataset = {
  if (dataset.isEmpty) {
    Vector.empty
  } else {
    val numberOfColumns = dataset.head.length
    val columnIndicesRange = (0 until numberOfColumns).toVector

    for {
      row <- dataset
    } yield {
      columnIndicesRange.map { columnIndex =>
        val rowData = row(columnIndex)

        minMaxes(columnIndex) match {
          case None => rowData
          case Some((min, max)) =>
            val rowValue = getNumericValue(rowData).get
            val normalizedRowValue = (rowValue - min) / (max - min)

            Numeric(normalizedRowValue)
        }
      }
    }
  }
}

def getColumnMeans(dataset: Dataset): StatisticData = {
  if (dataset.isEmpty) {
    Vector.empty
  } else {
    val numberOfColumns = dataset.head.length
    val testRow = dataset.head

    for {
      columnIndex <- (0 until numberOfColumns).toVector
    } yield {
      if (isText(testRow(columnIndex))) {
        None
      } else {
        val columnValues = dataset.map { row =>
          getNumericValue(row(columnIndex)).get
        }
        val sum = columnValues.sum
        val count = columnValues.length

        Some(sum / count)
      }
    }
  }
}

def getColumnsStandardDeviations(dataset: Dataset, means: StatisticData): StatisticData = {
  if (dataset.isEmpty) {
    Vector.empty
  } else {
    val numberOfColumns = dataset.head.length
    val testRow = dataset.head

    for {
      columnIndex <- (0 until numberOfColumns).toVector

    } yield {
      if (isText(testRow(columnIndex))) {
        None
      } else {
        val columnMean = means(columnIndex).get
        val columnSquaredMeanDifferences = dataset.map { row =>
          val meanDifference = getNumericValue(row(columnIndex)).get - columnMean

          math.pow(meanDifference, 2)
        }
        val sum = columnSquaredMeanDifferences.sum
        val count = columnSquaredMeanDifferences.length
        val variance = sum / (count - 1)
        val standardDeviation = math.sqrt(variance)

        Some(standardDeviation)
      }
    }
  }
}

def standardizeDataset(dataset: Dataset, means: StatisticData, standardDeviations: StatisticData): Dataset = {
  if (dataset.isEmpty) {
    Vector.empty
  } else {
    val numberOfColumns = dataset.head.length

    for {
      row <- dataset
      columnIndicesRange = (0 until numberOfColumns).toVector
    } yield {
      columnIndicesRange.map { columnIndex =>
        val rowData = row(columnIndex)

        if (isText(rowData)) {
          rowData
        } else {
          val columnMean = means(columnIndex).get
          val columnStandardDeviation = standardDeviations(columnIndex).get
          val rowValue = getNumericValue(rowData).get

          val standardizedRowValue = (rowValue - columnMean) / columnStandardDeviation

          Numeric(standardizedRowValue)
        }
      }
    }
  }
}



def trainTestSplit(dataset: Dataset, trainProportion: Double = 0.8, seed: Int = 42): (Dataset, Dataset) = {
  val numberOfRowsInTheTrainingSet = (dataset.length * trainProportion).toInt
  val random = new Random(seed)
  val shuffledDataset = random.shuffle(dataset)

  shuffledDataset.splitAt(numberOfRowsInTheTrainingSet)
}

def crossValidationSplit(dataset: Dataset, numberOfFolds: Int = 3, seed: Int = 42): Vector[Dataset] = {
  val foldSize = dataset.length / numberOfFolds
  val random = new Random(seed)
  val shuffledDataset = random.shuffle(dataset)

  shuffledDataset.grouped(foldSize).toVector
}

def f1(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Double = {
  assert(actual.length == predicted.length)

  val precisionValue = precision(actual, predicted, positiveLabel)
  val recallValue = recall(actual, predicted, positiveLabel)

  (precisionValue * recallValue) / (precisionValue + recallValue)
}

def recall(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Double = {
  assert(actual.length == predicted.length)

  val matrix = confusionMatrix(actual, predicted, positiveLabel)

  matrix("TP").toDouble / (matrix("TP") + matrix("FN")).toDouble
}


def precision(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Double = {
  assert(actual.length == predicted.length)

  val matrix = confusionMatrix(actual, predicted, positiveLabel)

  matrix("TP").toDouble / (matrix("TP") + matrix("FP")).toDouble
}

def rootMeanSquaredError(actual: Vector[Numeric], predicted: Vector[Numeric]): Double = {
  assert(actual.length == predicted.length)

  val sumOfSquaredErrors = actual.indices.foldLeft(0.0) { (accumulated, index) =>
    accumulated + math.pow(actual(index).value - predicted(index).value, 2)
  }

  math.sqrt(sumOfSquaredErrors / actual.length)
}

def meanAbsoluteError(actual: Vector[Numeric], predicted: Vector[Numeric]): Double = {
  assert(actual.length == predicted.length)

  val sumOfAbsoluteErrors = actual.indices.foldLeft(0.0) { (accumulated, index) =>
    accumulated + math.abs(actual(index).value - predicted(index).value)
  }

  sumOfAbsoluteErrors / actual.length
}

def confusionMatrix(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Map[String, Int] = {
  assert(actual.length == predicted.length)

  actual.indices.foldLeft(Map("TP" -> 0, "FP" -> 0, "FN" -> 0, "TN" -> 0)) { (matrix, index) =>
    val actualLabel = actual(index)
    val predictedLabel = predicted(index)

    if (actualLabel == positiveLabel) {
      if (actualLabel == predictedLabel) {
        matrix + ("TP" -> (matrix("TP") + 1))
      } else {
        matrix + ("FP" -> (matrix("FP") + 1))
      }
    } else {
      if (actualLabel == predictedLabel) {
        matrix + ("TN" -> (matrix("TN") + 1))
      } else {
        matrix + ("FN" -> (matrix("FN") + 1))
      }
    }
  }
}

def accuracy(actual: Vector[Data], predicted: Vector[Data]): Double = {
  assert(actual.length == predicted.length)

  val indices = actual.indices
  val numberOfTotalPredictions = predicted.length

  val numberOfCorrectPredictions = indices.foldLeft(0.0) { (accumulated, index) =>
    accumulated + (if (actual(index) == predicted(index)) 1.0 else 0.0)
  }

  numberOfCorrectPredictions / numberOfTotalPredictions
}

sealed trait Measure
case object Mean extends Measure
case object Mode extends Measure
case object Median extends Measure

def selectColumn(dataset: Dataset, index: Int): Vector[Data] = {
  dataset.map(_(index))
}

def randomAlgorithm(train: Dataset, test: Dataset, seed: Int = 42): Vector[Data] = {
  val random = new Random(seed)

  val outputColumn = selectColumn(train, train.head.length - 1)
  val uniqueOutputs = outputColumn.distinct
  val numberOfUniqueOutputs = uniqueOutputs.length

  test.map { row =>
    val randomIndex = random.nextInt(numberOfUniqueOutputs)

    uniqueOutputs(randomIndex)
  }
}

def zeroRuleClassifier(train: Dataset, test: Dataset): Vector[Data] = {
  val outputColumn = selectColumn(train, train.head.length - 1)

  val mode = outputColumn.groupBy(identity).maxBy(_._2.length)._1

  test.map(row => mode)
}

def zeroRuleRegressor(train: Dataset, test: Dataset, measure: Measure = Mean): Vector[Data] = {
  def calculateMean(labels: Vector[Data]) = Numeric {
    val sum = labels.foldLeft(0.0) { (accum, numericValue) => accum + getNumericValue(numericValue).get }

    sum / labels.length
  }

  def calculateMedian(labels: Vector[Data]) = {
    val sortedLabels = labels.sortBy(getNumericValue(_).get)
    val evenNumberOfLabels = labels.length % 2 == 0

    if (evenNumberOfLabels) {
      val splitIndex = labels.length / 2

      Numeric {
        (getNumericValue(sortedLabels(splitIndex - 1)).get + getNumericValue(sortedLabels(splitIndex)).get) /  2
      }
    } else {
      val medianIndex = labels.length / 2
      sortedLabels(medianIndex)
    }
  }

  def calculateMode(labels: Vector[Data]): Data = {
    labels.groupBy(identity).maxBy(_._2.length)._1
  }

  val outputColumn = selectColumn(train, train.head.length - 1)
  val measure = measure match {
    case Mean => calculateMean(outputColumn)
    case Mode => calculateMode(outputColumn)
    case Median => calculateMedian(outputColumn)
  }

  test.map(row => measure)
}

type Parameters = Map[String, Any]
type Output = Vector[Data]
type Algorithm = (Dataset, Dataset, Parameters) => Output
type EvaluationMetric[T <: Data] = (Vector[T], Vector[T]) => Double

def evaluateAlgorithmUsingTrainTestSplit[T <: Data](dataset: Dataset, algorithm: Algorithm, parameters: Parameters, evaluationMetric: EvaluationMetric[T], trainProportion: Double = 0.8, randomSeed: Int = 42): Unit = {
  val (train, test) = trainTestSplit(dataset, trainProportion, randomSeed)
  val predicted = algorithm(train, test, parameters)
  val actual = selectColumn(test, test.length - 1)

  evaluationMetric(actual, predicted)
}

def evaluateAlgorithmUsingCrossValidation[T <: Data](dataset: Dataset, algorithm: Algorithm, parameters: Parameters, evaluationMetric: EvaluationMetric[T], numberOfFolds: Int = 3, randomSeed: Int = 42) = {
  val folds = crossValidationSplit(dataset, numberOfFolds, randomSeed)

  for {
    fold <- folds
    train = folds.filterNot(_ == fold).flatten
    test = fold
  } yield {
    val predicted = algorithm(train, test, parameters)
    val actual = selectColumn(test, test.length - 1)

    evaluationMetric(actual, predicted)
  }
}

def mean(values: Vector[Numeric]): Double = values.foldLeft(0.0) { (accumulator: Double, numericValue: Numeric) =>
  accumulator + numericValue.value
} / values.length

def variance(values: Vector[Numeric], mean: Double): Double = values.foldLeft(0.0) { (accumulator: Double, numericValue: Numeric) =>
  accumulator + math.pow(numericValue.value - mean, 2)
}

def covariance(x: Vector[Numeric], y: Vector[Numeric], meanX: Double, meanY: Double): Double = {
  assert(x.length == y.length)

  x.indices.foldLeft(0.0) { (accumulator, index) =>
    accumulator + ((x(index).value - meanX) * (y(index).value - meanY))
  }
}

def coefficients(dataset: Dataset) = {
  val x = selectColumn(dataset, 0).map(d => Numeric(getNumericValue(d).get))
  val y = selectColumn(dataset, 1).map(d => Numeric(getNumericValue(d).get))

  val xMean = mean(x)
  val yMean = mean(y)

  val b1 = covariance(x, y, xMean, yMean) / variance(x, xMean)
  val b0 = yMean - b1 * xMean

  (b0, b1)
}


def simpleLinearRegression(train: Dataset, test: Dataset) = {
  val (b0, b1) = coefficients(train)

  test.map { case Vector(data, _) =>
    b0 + b1 * getNumericValue(data).get
  }
}

// Heads UP! Previous lines correspond to other scripts' contents.
// NEW content starts here:

def updatedVector[T](vector: Vector[T], newValue: T, index: Int) = {
  val (firstHalf, secondHalf) = vector.splitAt(index)
  firstHalf ++ Vector(newValue) ++ secondHalf.tail
}

def predictLinearRegression(row: Vector[Data], coefficients: Vector[Double]): Double = {
  val indices = row.indices.init

  indices.foldLeft(0.0) { (accumulator, index) =>
    accumulator + coefficients(index + 1) * getNumericValue(row(index)).get
  } + coefficients.head
}

def coefficientsLinearRegressionSgd(train: Dataset, learningRate: Double, numberOfEpochs: Int) = {
  var coefficients = Vector.fill(train.head.length)(0.0)

  for {
    _ <- 1 to numberOfEpochs
    row <- train
    predicted = predictLinearRegression(row, coefficients)
    actual = getNumericValue(row.last).get
    error = predicted - actual
  } {
    // TODO Bias?
    val firstCoefficient = coefficients.head - learningRate * error
    val indices = row.indices.init

    val remainingCoefficients = indices.foldLeft(coefficients) { (c, index) =>
      updatedVector(c, c(index + 1) - learningRate * error * getNumericValue(row(index)).get, index + 1)
    }

    coefficients = Vector(firstCoefficient) ++ remainingCoefficients
  }

  coefficients
}

def linearRegressionSgd(train: Dataset, test: Dataset, parameters: Parameters) = {
  val learningRate = parameters("learningRate").asInstanceOf[Double]
  val numberOfEpochs = parameters("numberOfEpochs").asInstanceOf[Int]

  val coefficients = coefficientsLinearRegressionSgd(train, learningRate, numberOfEpochs)

  test.map { row =>
    predictLinearRegression(row, coefficients)
  }
}