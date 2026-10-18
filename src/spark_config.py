from pyspark.sql import SparkSession

def get_spark_session():
    spark = SparkSession.builder \
        .master(local[*])
        .appName("ResearchNavigator") \
        .config("spark.driver.memory", "5g") \
        .config("spark.executor.memory", "5g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark