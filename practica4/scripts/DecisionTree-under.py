import sys

from pyspark import SparkContext, SparkConf, SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == '__main__':

    #Creacion del contexto de Spark
    conf = SparkConf().setAppName('Angel Murcia Diaz -> Practica 4 - Decision Tree - Dataset Blanceado con Undersampling')  
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

    #Balanceo de los datos utilizando UnderSampling
    NoDataset = dataset.filter('label=0')
    SiDataset = dataset.filter('label=1')
    #print('total, 1.0, 0.0 = ', dataset.count(), NoDataset.count(), SiDataset.count())
    sampleRatio = float(SiDataset.count()) / float(dataset.count())
    seleccion = NoDataset.sample(False, sampleRatio)
    dataset = SiDataset.unionAll(seleccion)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(dataset)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol='features', outputCol='indexedFeatures', maxCategories=2).fit(dataset)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures')

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select('prediction', 'indexedLabel', 'features').show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol='indexedLabel', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Test Error = %g ' % (1.0 - accuracy))
    print('Accuracy = ', accuracy)

    treeModel = model.stages[2]
    # summary only
    print(treeModel)

    #Calcular AUC
    evaluator = BinaryClassificationEvaluator()
    evaluation = evaluator.evaluate(model.transform(testData))
    print('AUC:', evaluation)

    #Detener
    sc.stop()