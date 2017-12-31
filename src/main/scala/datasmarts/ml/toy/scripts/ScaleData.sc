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

    for {
      columnIndex <- (0 until numberOfColumns).toVector
      testRow = dataset.head
    } yield {
      if (isText(testRow(columnIndex))) {
        None
      } else {
        val columnValues = dataset.map(r => getNumericValue(r(columnIndex)).get).sorted
        val max = columnValues.last
        val min = columnValues.head

        Some((min, max))
      }
    }
  }
}

getDatasetMinAndMax(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("B")),
))

def normalizeDataset(dataset: Dataset, minMaxes: MinMaxData): Dataset = {
  val numberOfColumns = dataset.head.length

  for {
    row <- dataset
    columnIndicesRange = (0 until numberOfColumns).toVector
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

normalizeDataset(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("B")),
), getDatasetMinAndMax(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("B")),
)))

def getColumnMeans(dataset: Dataset): StatisticData = {
  if (dataset.isEmpty) {
    Vector.empty
  } else {
    val numberOfColumns = dataset.head.length

    for {
      columnIndex <- (0 until numberOfColumns).toVector
      testRow = dataset.head
    } yield {
      if (isText(testRow(columnIndex))) {
        None
      } else {
        val columnValues = dataset.map(r => getNumericValue(r(columnIndex)).get)
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

    for {
      columnIndex <- (0 until numberOfColumns).toVector
      testRow = dataset.head
    } yield {
      if (isText(testRow(columnIndex))) {
        None
      } else {
        val columnMean = means(columnIndex).get
        val columnSquaredMeanDifferences = dataset.map(r => math.pow(getNumericValue(r(columnIndex)).get - columnMean, 2))
        val sum = columnSquaredMeanDifferences.sum
        val count = columnSquaredMeanDifferences.length

        Some(math.sqrt(sum / (count - 1)))
      }
    }
  }
}

def standardizeDataset(dataset: Dataset, means: StatisticData, standardDeviations: StatisticData): Dataset = {
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

standardizeDataset(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("C")),
), getColumnMeans(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("C")),
)), getColumnsStandardDeviations(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("C")),
), getColumnMeans(Vector(
  Vector(Numeric(50), Numeric(30), Text("A")),
  Vector(Numeric(20), Numeric(90), Text("B")),
  Vector(Numeric(19), Numeric(90.4), Text("C")),
))))

// HEADS UP! --> LoadCsv.sc contents replicated here due to Scala Worksheets limitation of importing other worksheet.s

import com.github.tototoshi.csv.CSVReader

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