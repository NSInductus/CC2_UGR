import sys

from pyspark import SparkContext, SparkConf, SQLContext

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.mllib.evaluation import BinaryClassificationMetrics

if __name__ == '__main__':

    #Creacion del contexto de Spark
    conf = SparkConf().setAppName('Angel Murcia Diaz -> Practica 4 - MLP - Dataset Balanceado con UnderSamplig - Topologia 6 6 4 2')  
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

    # Split the data into train and test
    splits = dataset.randomSplit([0.7, 0.3], 1234)
    train = splits[0]
    test = splits[1]

    # specify layers for the neural network:
    # input layer of size 6 (features), two intermediate of size 6 and 4
    # and output of size 2 (classes)
    layers = [6, 6, 4, 2]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

    # train the model
    model = trainer.fit(train)

    # compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select('prediction', 'label')
    evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
    print('Test set accuracy = ' + str(evaluator.evaluate(predictionAndLabels)))

    #Calcular AUC
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction')
    evaluation = evaluator.evaluate(model.transform(test))
    print('AUC:', evaluation)

    #Detener
    sc.stop()