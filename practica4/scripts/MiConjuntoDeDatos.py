import sys
 
from pyspark import SparkContext, SparkConf, SQLContext
 
if __name__ == '__main__':
 
  # create Spark context with Spark configuration
  conf = SparkConf().setAppName('Angel Murcia Diaz -> Practica 4 - Primeros Pasos')  

  sc = SparkContext(conf=conf)
 
  #df = sc.read.csv('/user/ccsa26821637/completo.csv',header=True,sep=',',inferSchema=True)
 
  #df.show()

  sqlc = SQLContext(sc)
  df = sqlc.read.csv('/user/ccsa26821637/completo.csv', header=True, sep=',',inferSchema=True)
 
  #df.createOrReplaceTempView("sql_dataset")
 
  #sqlDF = sc.sql('SELECT PSSM_r1_1_M, PSSM_central_2_N, PSSM_central_-1_S, PSSM_r2_-2_W, PSSM_central_-1_R, AA_freq_global_R, class FROM sql_dataset')
  #sqlDF.show()

  # Get the assigned columns:
  df = df.select('PSSM_r1_1_M', 'PSSM_central_2_N', 'PSSM_central_-1_S', 'PSSM_r2_-2_W', 'PSSM_central_-1_R', 'AA_freq_global_R', 'class')


  # Write new csv:
  df.write.csv('/user/ccsa26821637/filteredC.small.training', header=True)

  sc.stop()

