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

def getColumnsMeans(dataset: Dataset): StatisticData = {
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

def precision(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Double = {
  assert(actual.length == predicted.length)

  val matrix = confusionMatrix(actual, predicted, positiveLabel)

  matrix("TP").toDouble / (matrix("TP") + matrix("FP")).toDouble
}

def recall(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Double = {
  assert(actual.length == predicted.length)

  val matrix = confusionMatrix(actual, predicted, positiveLabel)

  matrix("TP").toDouble / (matrix("TP") + matrix("FN")).toDouble
}

def f1(actual: Vector[Data], predicted: Vector[Data], positiveLabel: Data): Double = {
  assert(actual.length == predicted.length)

  val precisionValue = precision(actual, predicted, positiveLabel)
  val recallValue = recall(actual, predicted, positiveLabel)

  (precisionValue * recallValue) / (precisionValue + recallValue)
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
  val measureValue = measure match {
    case Mean => calculateMean(outputColumn)
    case Mode => calculateMode(outputColumn)
    case Median => calculateMedian(outputColumn)
  }

  test.map(_ => measureValue)
}

type Parameters = Map[String, Any]
type Algorithm = (Dataset, Dataset, Parameters) => Vector[Data]
type EvaluationMetric[T <: Data] = (Vector[T], Vector[T]) => Double

def evaluateAlgorithmUsingTrainTestSplit[T <: Data](dataset: Dataset, algorithm: Algorithm, parameters: Parameters, evaluationMetric: EvaluationMetric[T], trainProportion: Double = 0.8, randomSeed: Int = 42) = {
  val (train, test) = trainTestSplit(dataset, trainProportion, randomSeed)
  val predicted = algorithm(train, test, parameters).asInstanceOf[Vector[T]]
  val actual = selectColumn(test, test.head.length - 1).asInstanceOf[Vector[T]]

  evaluationMetric(actual, predicted)
}

def evaluateAlgorithmUsingCrossValidation[T <: Data](dataset: Dataset, algorithm: Algorithm, parameters: Parameters, evaluationMetric: EvaluationMetric[T], numberOfFolds: Int = 3, randomSeed: Int = 42) = {
  val folds = crossValidationSplit(dataset, numberOfFolds, randomSeed)

  for {
    fold <- folds
    train = folds.filterNot(_ == fold).flatten
    test = fold
  } yield {
    val predicted = algorithm(train, test, parameters).asInstanceOf[Vector[T]]
    val actual = selectColumn(test, test.head.length - 1).asInstanceOf[Vector[T]]

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

def weights(dataset: Dataset) = {
  val x = selectColumn(dataset, 0).map(d => Numeric(getNumericValue(d).get))
  val y = selectColumn(dataset, 1).map(d => Numeric(getNumericValue(d).get))

  val xMean = mean(x)
  val yMean = mean(y)

  val b1 = covariance(x, y, xMean, yMean) / variance(x, xMean)
  val b0 = yMean - b1 * xMean

  (b0, b1)
}


def simpleLinearRegression(train: Dataset, test: Dataset) = {
  val (b0, b1) = weights(train)

  test.map { case Vector(data, _) =>
    b0 + b1 * getNumericValue(data).get
  }
}

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
    Numeric(predictLinearRegression(row, coefficients))
  }
}

def predictLogisticRegression(row: Vector[Data], coefficients: Vector[Double]): Double = {
  val indices = row.indices.init

  val yHat = indices.foldLeft(0.0) { (accumulator, index) =>
    accumulator + coefficients(index + 1) * getNumericValue(row(index)).get
  } + coefficients.head

  1.0 / (1.0 + math.exp(-yHat))
}

def coefficientsLogisticRegressionSgd(train: Dataset, learningRate: Double, numberOfEpochs: Int) = {
  var coefficients = Vector.fill(train.head.length)(0.0)

  for {
    _ <- 1 to numberOfEpochs
    row <- train
    predicted = predictLogisticRegression(row, coefficients)
    actual = getNumericValue(row.last).get
    error = predicted - actual
  } {
    // TODO Bias?
    val firstCoefficient = coefficients.head + learningRate * error * predicted * (1.0 - predicted)
    val indices = row.indices.init

    val remainingCoefficients = indices.foldLeft(coefficients) { (c, index) =>
      val actual = getNumericValue(row(index)).get
      updatedVector(c, c(index + 1) + learningRate * error * predicted * (1.0 - predicted) * actual, index + 1)
    }

    coefficients = Vector(firstCoefficient) ++ remainingCoefficients
  }

  coefficients
}

def logisticRegression(train: Dataset, test: Dataset, parameters: Parameters) = {
  val learningRate = parameters("learningRate").asInstanceOf[Double]
  val numberOfEpochs = parameters("numberOfEpochs").asInstanceOf[Int]

  val coefficients = coefficientsLogisticRegressionSgd(train, learningRate, numberOfEpochs)

  test.map { row =>
    Numeric(math.round(predictLogisticRegression(row, coefficients)))
  }
}

def predictWithWeights(row: Vector[Data], weights: Vector[Double]) = {
  val indices = row.indices.init

  val activation = indices.foldLeft(0.0) { (accumulator, index) =>
    accumulator + weights(index + 1) * getNumericValue(row(index)).get
  } + weights.head

  if (activation >= 0.0) 1.0 else 0.0
}

def trainWeights(train: Dataset, learningRate: Double, numberOfEpochs: Int) = {
  var weights = Vector.fill(train.head.length)(0.0)

  for {
    _ <- 1 to numberOfEpochs
    row <- train
  } {

    val predicted = predictWithWeights(row, weights)
    val actual = getNumericValue(row.last).get
    val error = predicted - actual

    val bias = weights.head + learningRate * error
    val indices = row.indices.init

    val remainingWeights = indices.foldLeft(weights) { (w, index) =>
      val actual = getNumericValue(row(index)).get
      updatedVector(w, w(index + 1) + learningRate * error * actual, index + 1)
    }

    weights = Vector(bias) ++ remainingWeights.tail
  }

  weights
}

def perceptron(train: Dataset, test: Dataset, parameters: Parameters) = {
  val learningRate = parameters("learningRate").asInstanceOf[Double]
  val numberOfEpochs = parameters("numberOfEpochs").asInstanceOf[Int]

  val weights = trainWeights(train, learningRate, numberOfEpochs)

  test.map { row =>
    Numeric(predictWithWeights(row, weights))
  }
}

def separateByClass(dataset: Dataset): Map[Data, Vector[Vector[Data]]] = {
  dataset.groupBy(_.last)
}

def summarizeDataset(dataset: Dataset) = {
  val numberOfColumns = dataset.head.length

  val means = getColumnsMeans(dataset)
  val standardDeviations = getColumnsStandardDeviations(dataset, means)
  val counts = (1 to dataset.head.length).toVector.map(_ => dataset.length)

  assert(List(means.length, standardDeviations.length, counts.length).forall(_ == numberOfColumns))

  // We ignore the labels column
  (0 until numberOfColumns - 1).toVector.map(i => (means(i).get, standardDeviations(i).get, counts(i)))
}

def summarizeByClass(dataset: Dataset) = {
  val separated = separateByClass(dataset)

  separated.mapValues(summarizeDataset)
}

def calculateProbability(x: Double, mean: Double, standardDeviation: Double) = {
  val exponent = math.exp(-(math.pow(x - mean, 2) / (2 * standardDeviation * standardDeviation)))
  (1.0 / (math.sqrt(2 * math.Pi) * standardDeviation)) * exponent
}

def calculateClassProbabilities(summaries: Map[Data, Vector[(Double, Double, Int)]], row: Vector[Data]) = {
  val totalRows = summaries.foldLeft(0){ (accum, entry) =>
    entry match {
      case (_, summary) => accum + summary.head._3
    }
  }

  summaries.mapValues { summaries =>
    var a = summaries.head._3 / totalRows.toDouble

    // Class probability
    summaries.indices.foldLeft(summaries.head._3 / totalRows.toDouble) { (classProbability, i) =>
      val (mean, standardDeviation, _) = summaries(i)
      classProbability * calculateProbability(getNumericValue(row(i)).get, mean, standardDeviation)
    }
  }
}

def predict(summaries: Map[Data, Vector[(Double, Double, Int)]], row: Vector[Data]): Data = {
  val probabilities = calculateClassProbabilities(summaries, row)

  val (Some(bestLabel), _) = probabilities.foldLeft((None: Option[Data], -1.0)) { (bestLabelAndProb, entry) =>
    entry match {
      case (label, classProbability) =>
        val (bestLabel, bestProbability) = bestLabelAndProb

        if (bestLabel.isEmpty || classProbability > bestProbability) {
          (Some(label), classProbability)
        } else {
          bestLabelAndProb
        }
    }
  }

  bestLabel
}

def naiveBayes(train: Dataset, test: Dataset) = {
  val summaries = summarizeByClass(train)

  test.map { row =>
    predict(summaries, row)
  }
}

def euclideanDistance(firstRow: Vector[Numeric], secondRow: Vector[Numeric]) = {
  assert(firstRow.length == secondRow.length)

  math.sqrt {
    val featureIndices = firstRow.indices.init

    featureIndices.foldLeft(0.0) { (accum, i) =>
      accum + math.pow(firstRow(i).value - secondRow(i).value, 2)
    }
  }
}

def getNeighbors(train: Dataset, testRow: Vector[Numeric], numberOfNeighbors: Int) = {
  val neighborsAndDistances = for {
    row <- train
    numericRow = row.asInstanceOf[Vector[Numeric]]
  } yield {
    val distance = euclideanDistance(numericRow, testRow)
    (numericRow, distance)
  }

  neighborsAndDistances.sortBy(_._2).take(numberOfNeighbors).map(_._1)
}

def predictClassification(train: Dataset, testRow: Vector[Numeric], numberOfNeighbors: Int) = {
  val neighbors = getNeighbors(train, testRow, numberOfNeighbors)
  val outputValues = neighbors.map(_.last)

  outputValues.maxBy(o => outputValues.count(_ == o))
}

def predictRegression(train: Dataset, testRow: Vector[Numeric], numberOfNeighbors: Int) = {
  val neighbors = getNeighbors(train, testRow, numberOfNeighbors)
  val outputValues = neighbors.map(_.last)

  Numeric {
    outputValues.foldLeft(0.0) { (total, numeric) => total + numeric.value } / outputValues.length
  }
}

type Predictor = (Dataset, Vector[Numeric], Int) => Numeric
def kNearestNeighbors(train: Dataset, test: Dataset, parameters: Parameters) = {
  val numberOfNeighbors = parameters("numberOfNeighbors").asInstanceOf[Int]
  val predictor = parameters("predictor").asInstanceOf[Predictor]

  test.map { row =>
    predictor(train, row.asInstanceOf[Vector[Numeric]], numberOfNeighbors)
  }
}

def getBestMatchingUnit(codebooks: Vector[Vector[Numeric]], testRow: Vector[Numeric]) = {
  val codebooksDistances = for {
    codebook <- codebooks
  } yield {
    val distance = euclideanDistance(codebook, testRow)
    (codebook, distance)
  }

  codebooksDistances.minBy(_._2)._1
}

def randomCodebook(train: Dataset) = {
  val numberOfRecords = train.length
  val numberOfFeatures = train.head.length

  (0 until numberOfFeatures).map { index =>
    train(Random.nextInt(numberOfRecords))(index)
  }.toVector
}

def trainCodebooks(train: Dataset, numberOfCodebooks: Int, learningRate: Double, numberOfEpochs: Int) = {
  var codebooks = (0 until numberOfCodebooks).map(_ => randomCodebook(train).asInstanceOf[Vector[Numeric]]).toVector

  for (epoch <- 0 until numberOfEpochs) {
    val rate = learningRate * (1.0 - (epoch / numberOfEpochs))

    for (row <- train) {
      val numericRow = row.asInstanceOf[Vector[Numeric]]
      var bestMatchingUnit = getBestMatchingUnit(codebooks, numericRow)
      val bestMatchingUnitIndex = codebooks.indexOf(bestMatchingUnit)

      val rowFeaturesIndices = row.indices.take(row.length - 2)
      rowFeaturesIndices.foreach { i =>
        val error = numericRow(i).value - bestMatchingUnit(i).value
        val updatedValue = Numeric {
          if (bestMatchingUnit.last == numericRow.last) {
            bestMatchingUnit(i).value + error * rate
          } else {
            bestMatchingUnit(i).value - error * rate
          }
        }

        bestMatchingUnit = updatedVector(bestMatchingUnit, updatedValue, i)
      }

      codebooks = updatedVector(codebooks, bestMatchingUnit, bestMatchingUnitIndex)
    }
  }

  codebooks
}

def predictWithCodebooks(codebooks: Vector[Vector[Numeric]], testRow: Vector[Numeric]): Numeric = {
  getBestMatchingUnit(codebooks, testRow).last
}

def learningVectorQuantization(train: Dataset, test: Dataset, numberOfCodebooks: Int, learningRate: Double, numberOfEpochs: Int) = {
  val codebooks = trainCodebooks(train, numberOfCodebooks, learningRate, numberOfEpochs)

  test.map { row =>
    predictWithCodebooks(codebooks, row.asInstanceOf[Vector[Numeric]])
  }
}

// Heads UP! Previous lines correspond to other scripts' contents.
// NEW content starts here:

sealed trait TreeNode
type Group = Vector[Vector[Data]]
case class InnerNode(index: Int, value: Double, groups: Option[Vector[Group]] = None, left: Option[TreeNode] = None, right: Option[TreeNode] = None) extends TreeNode
case class TerminalNode(outcome: Numeric) extends TreeNode

def testSplit(index: Int, value: Numeric, dataset: Dataset): Vector[Vector[Vector[Data]]] = {
  val numericValue = value.value
  val (left, right) = dataset.partition(r => getNumericValue(r(index)).get < numericValue)

  Vector(left, right)
}

def giniIndex(groups: Vector[Group], classValues: Vector[Numeric]) = {
  classValues.foldLeft(0.0) { (accumulatedGini, classValue) =>
    groups.foldLeft(accumulatedGini) { (g, group) =>
      if (group.nonEmpty) {
        val proportion = group.map(_.last).count(_ == classValue).toDouble / group.length.toDouble
        g + proportion * (1.0 - proportion)
      } else {
        g
      }
    }
  }
}

def getSplit(dataset: Dataset) = {
  val classValues = selectColumn(dataset, dataset.head.length - 1).distinct

  var bIndex = 999
  var bValue = 999.0
  var bScore = 999.0
  var bGroups = Vector.empty[Group]

  for (index <- 0 until (dataset.head.length - 1)) {
    for (row <- dataset) {
      val groups = testSplit(index, row(index).asInstanceOf[Numeric], dataset)
      val gini = giniIndex(groups, classValues.asInstanceOf[Vector[Numeric]])

      if (gini < bScore) {
        bIndex = index
        bValue = row(index).asInstanceOf[Numeric].value
        bScore = gini
        bGroups = groups
      }
    }
  }

  InnerNode(index = bIndex, value = bValue, groups = Some(bGroups))
}

def toTerminal(group: Group): TerminalNode = {
  val outcomes = group.map(_.last)
  TerminalNode {
    outcomes.maxBy(o => outcomes.count(_ == o)).asInstanceOf[Numeric]
  }
}

def split(node: TreeNode, maxDepth: Int, minSize: Int, depth: Int): TreeNode = {
  node match {
    case terminalNode: TerminalNode =>
      terminalNode

    case innerNode @ InnerNode(index, value, Some(Vector(leftGroup, rightGroup)), _, _)
      if leftGroup.isEmpty || rightGroup.isEmpty =>

      val t = toTerminal(leftGroup ++ rightGroup)
      innerNode.copy(left = Some(t), right = Some(t))

    case innerNode: InnerNode if innerNode.groups.isDefined =>
      val groups: Vector[Group] = innerNode.groups.get
      val Vector(left, right) = groups

      if (depth >= maxDepth) {
        innerNode.copy(left = Some(toTerminal(left)), right = Some(toTerminal(right)))
      } else {
        val leftNode = if (left.lengthCompare(minSize) <= 0) {
          innerNode.copy(left = Some(toTerminal(left)))
        } else {
          val n = innerNode.copy(left = Some(getSplit(left)))
          split(n, maxDepth, minSize, depth + 1)
        }

        val rightNode = if (right.lengthCompare(minSize) <= 0) {
          innerNode.copy(right = Some(toTerminal(right)))
        } else {
          val n = innerNode.copy(right = Some(getSplit(right)))
          split(n, maxDepth, minSize, depth + 1)
        }

        innerNode.copy(left = Some(leftNode), right = Some(rightNode))
      }
  }
}

def buildTree(train: Dataset, maxDepth: Int, minSize: Int) = split(getSplit(train), maxDepth, minSize, 1)


def predictWithTree(node: TreeNode, row: Vector[Numeric]): Numeric = node match {
  case TerminalNode(outcome) => outcome
  case i: InnerNode =>
    if (getNumericValue(row(i.index)).get <= i.value) {
      predictWithTree(i.left.get, row)
    } else {
      predictWithTree(i.right.get, row)
    }
}


def decisionTree(train: Dataset, test: Dataset, parameters: Parameters) = {
  val maxDepth = parameters("maxDepth").asInstanceOf[Int]
  val minSize = parameters("minSize").asInstanceOf[Int]

  val t = buildTree(train, maxDepth, minSize)

  test.map { row =>
    predictWithTree(t, row.asInstanceOf[Vector[Numeric]])
  }
}



// Heads UP! Previous lines correspond to other scripts' contents.
// NEW content starts here:

type Layer = Vector[Map[String, Any]]
type Network = Vector[Layer]

def initializeNetwork(numberOfInputs: Int, numberOfHiddenUnits: Int, numberOfOutputs: Int): Network = {
  val hiddenLayer = (0 until numberOfHiddenUnits).map { _ =>
    Map("weights" -> (0 to numberOfInputs).map(_ => Random.nextDouble()).toVector)
  }.toVector

  val outputLayer = (0 until numberOfOutputs).map { _ =>
    Map("weights" -> (0 to numberOfHiddenUnits).map(_ => Random.nextDouble()).toVector)
  }.toVector

  Vector(hiddenLayer, outputLayer)
}

initializeNetwork(2, 1, 2).foreach(println)

def activate(weights: Vector[Double], inputs: Vector[Double]) = {
  val activation = weights.last

  weights.indices.init.foldLeft(activation) { (accum, i) =>
    accum + weights(i) * inputs(i)
  }
}

def transfer(activation: Double) = {
  1.0 / (1.0 + math.exp(-activation))
}

// TODO Modify this
def forwardPropagate(network: Network, row: Vector[Numeric]): (Vector[Double], Network) = {
  var inputs = row.map(_.value)
  var updatedNetwork = network

  for {
    i <- network.indices
  } {
    var layer = updatedNetwork(i)
    val newInputs = layer.indices.map { i =>
      var neuron = layer(i)
      val activation = activate(neuron("weights").asInstanceOf[Vector[Double]], inputs)
      neuron = neuron + ("output" -> transfer(activation))
      layer = updatedVector(layer, neuron, i)

      neuron("output").asInstanceOf[Double]
    }

    inputs = newInputs.toVector

    updatedNetwork = updatedVector(updatedNetwork, layer, i)
  }

  (inputs, updatedNetwork)
}

val network: Network = Vector(
  Vector(
    Map("weights" -> Vector(0.864973644875332, 0.4299962390096266, 0.25159829718468507)),
    Map("weights" -> Vector(0.8440020790298866, 0.18310892263206968, 0.34381820668252616))),
  Vector(
    Map("weights" -> Vector(0.7251975317621959, 0.06827255520222186))))

forwardPropagate(network, Vector(Numeric(1.0), Numeric(0.0), Numeric(1.0)))

def transferDerivative(output: Double) = {
  output * (1.0 - output)
}

def backwardPropagateError(network: Network, expected: Vector[Double]) = {
  var updatedNetwork = network
  for {
    i <- (updatedNetwork.length - 1) to 0 by -1
    isOutputLayer = i == updatedNetwork.indices.last
  } {
    val layer = updatedNetwork(i)
    var errors = Vector.empty[Double]

    if (isOutputLayer) {
      errors = layer.indices.map(i => expected(i) - layer(i)("output").asInstanceOf[Double]).toVector
    } else {
      errors = layer.indices.map { j =>
        val error = updatedNetwork(i + 1).foldLeft(0.0) { (e, neuron) =>
          e + neuron("weights").asInstanceOf[Vector[Double]](j) * neuron("delta").asInstanceOf[Double]
        }

        error
      }.toVector
    }

    val newLayer = layer.indices.toVector.map { j =>
      var neuron = layer(j)
      neuron = neuron + ("delta" -> errors(j) * transferDerivative(neuron("output").asInstanceOf[Double]))
      neuron
    }

    updatedNetwork = updatedVector(updatedNetwork, newLayer, i)
  }

  updatedNetwork
}

backwardPropagateError(forwardPropagate(network, Vector(Numeric(1.0), Numeric(0.0), Numeric(1.0)))._2, Vector(1.0, 0.0))

def updateWeights(network: Network, row: Vector[Numeric], learningRate: Double) = {
  var updatedNetwork = network

  for {
    i <- updatedNetwork.indices
    isFirstLayer = i == 0
  } {
    var inputs = row.init.map(_.value)

    if (!isFirstLayer) {
      inputs = network(i - 1).map(_("output").asInstanceOf[Double])
    }

    var newLayer = updatedNetwork(i)

    for {
      n <- newLayer.indices
    } {
      var newN: Map[String, Any] = updatedNetwork(i)(n)
      val delta = newN("delta").asInstanceOf[Double]
      val ws = newN("weights").asInstanceOf[Vector[Double]]
      var x = inputs.indices.foldLeft(ws) { (w, i) =>
        updatedVector(w, w(i) + learningRate * delta * inputs(i), i)
      }

      x = updatedVector(x, learningRate * delta + x(x.indices.last), x.indices.last)

      newN = newN + ("weights" -> x)

      newLayer = updatedVector(newLayer, newN, n)
    }

    updatedNetwork = updatedVector(updatedNetwork, newLayer, i)
  }

  updatedNetwork
}

def trainNetwork(network: Network, train: Dataset, learningRate: Double, numberOfEpochs: Int, numberOfOutputs: Int) = {
  var updatedNetwork = network
  for (e <- 1 to numberOfEpochs) {
    for (r <- train) {
      val (_, nn) = forwardPropagate(updatedNetwork, r.asInstanceOf[Vector[Numeric]])
      updatedNetwork = nn

      val expected = updatedVector((1 to numberOfOutputs).toVector.map(_ => 0.0), 1.0, r.last.asInstanceOf[Numeric].value.toInt)

      updatedNetwork = backwardPropagateError(updatedNetwork, expected)
      updatedNetwork = updateWeights(updatedNetwork, r.asInstanceOf[Vector[Numeric]], learningRate)
    }
  }

  updatedNetwork
}

val mockDataset = Vector((2.7810836,2.550537003,0),
  (1.465489372,2.362125076,0),
  (3.396561688,4.400293529,0),
  (1.38807019,1.850220317,0),
  (3.06407232,3.005305973,0),
  (7.627531214,2.759262235,1),
  (5.332441248,2.088626775,1),
  (6.922596716,1.77106367,1),
  (8.675418651,-0.242068655,1),
  (7.673756466,3.508563011,1)).map { case (x1, x2, y) => Vector(Numeric(x1), Numeric(x2), Numeric(y)) }

val n = trainNetwork(initializeNetwork(2 ,2, 2), mockDataset, 0.5, 20, 2)

def predictWithNetwork(network: Network, row: Vector[Numeric]) = {
  val (outputs, _) = forwardPropagate(network, row)
  outputs.indices.maxBy(outputs)
}

for (r <- mockDataset) {
  val prediction = predictWithNetwork(n, r.asInstanceOf[Vector[Numeric]])
  println(s"Expected=${r.last.value}, Predicted=$prediction")
}

def backPropagation(train: Dataset, test: Dataset, parameters: Parameters) = {
  val numberOfInputs = train.head.length - 1
  val numberOfOutputs = train.map(_.last).distinct.length
  val learningRate = parameters("learningRate").asInstanceOf[Double]
  val numberOfEpochs = parameters("numberOfEpochs").asInstanceOf[Int]
  val numberOfHiddenUnits = parameters("numberOfHiddenUnits").asInstanceOf[Int]

  val n: Network = trainNetwork(initializeNetwork(numberOfInputs, numberOfHiddenUnits, numberOfOutputs), train, learningRate, numberOfEpochs, numberOfOutputs)

  test.map { r =>
    Numeric(predictWithNetwork(n, r.asInstanceOf[Vector[Numeric]]))
  }
}

val banknotePath = "/media/jesus/ADATA HV100/Blog/toy-ml/src/main/resources/data/15/seeds_dataset.csv"

val rawData = loadCsv(banknotePath)
val numberOfRows = rawData.length
val numberOfColumns = rawData.head.length
println(s"Number of rows in dataset: $numberOfRows")
println(s"Number of columns in dataset: $numberOfColumns")

val (data, lookUpTable) = {
  val dataWithNumericColumns = (0 until (numberOfColumns - 1)).toVector.foldLeft(rawData) { (d, i) => textColumnToNumeric(d, i) }
  categoricalColumnToNumeric(dataWithNumericColumns, numberOfColumns - 1)
}

val baselineAccuracy = evaluateAlgorithmUsingTrainTestSplit[Numeric](
  data,
  (train, test, parameters) => zeroRuleClassifier(train, test),
  Map.empty,
  accuracy,
  trainProportion=0.8)

println(s"Zero Rule Algorithm accuracy: $baselineAccuracy")


val backPropagationAccuracy = evaluateAlgorithmUsingTrainTestSplit[Numeric](
  data,
  backPropagation,
  Map("learningRate" -> 0.3, "numberOfEpochs" -> 500, "numberOfHiddenUnits" -> 5),
  accuracy,
  trainProportion=0.8)

println(s"Back Propagation accuracy: $backPropagationAccuracy")