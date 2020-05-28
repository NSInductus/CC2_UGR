#Carga de librerias necesarias
import sys

from pyspark import SparkContext, SparkConf, SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Programa principal
if __name__ == '__main__':

    #Creacion del contexto de Spark
    conf = SparkConf().setAppName('Angel Murcia Diaz -> Practica 4 - Random Forest - Dataset Desbalanceado - 10 arboles')  
    sc = SparkContext(conf=conf)

    #Creacion del contexto SQL
    sqlc = SQLContext(sc)
    df = sqlc.read.csv('/user/ccsa26821637/filteredC.small.training', header=True, sep=',',inferSchema=True)

    #Conversion del conjunto de datos a formato legible
    assembler = VectorAssembler(inputCols=['PSSM_r1_1_M', 'PSSM_central_2_N', 'PSSM_central_-1_S', 'PSSM_r2_-2_W', 'PSSM_central_-1_R', 'AA_freq_global_R'], outputCol='features')
    dataset = assembler.transform(df)
    dataset = dataset.selectExpr('features as features', 'class as label')
    dataset = dataset.select('features', 'label')
    #dataset.show()

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(dataset)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
    VectorIndexer(inputCol='features', outputCol='indexedFeatures', maxCategories=2).fit(dataset)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures', numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol='prediction', outputCol='predictedLabel',
                labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select('predictedLabel', 'label', 'features').show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
    labelCol='indexedLabel', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Test Error = %g' % (1.0 - accuracy))
    print('Accuracy = ', accuracy)

    rfModel = model.stages[2]
    print(rfModel)  # summary only

    #Calcular AUC
    evaluator = BinaryClassificationEvaluator()
    evaluation = evaluator.evaluate(model.transform(testData))
    print('AUC:', evaluation)

    #Detener
    sc.stop()