from io import StringIO
import tempfile
import csv
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.functions import col
import os


if __name__ == "__main__":
        # Create a SparkSession
        spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

        '''
        # Get the raw data
        with open(temp_file.name, 'r') as f:
            lines = csv.reader(f)
        # Convert it to a RDD of Row objects with (userID, age, gender, occupation, zip)
        users = lines.map(parseInput)
        # Convert that to a DataFrame
        usersDataset = spark.createDataFrame(users)
        '''

        # Define the schema of the CSV file
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", IntegerType(), True),
            StructField("cp", IntegerType(), True),
            StructField("trestbps", IntegerType(), True),
            StructField("chol", IntegerType(), True),
            StructField("fbs", IntegerType(), True),
            StructField("restecg", IntegerType(), True),
            StructField("thalach", IntegerType(), True),
            StructField("exang", IntegerType(), True),
            StructField("oldpeak", DoubleType(), True),
            StructField("slope", IntegerType(), True),
            StructField("ca", IntegerType(), True),
            StructField("thal", IntegerType(), True),
            StructField("heartdisease", IntegerType(), True)
        ])

        # Read the CSV file using the specified schema
        usersDataset = spark.read.format("csv")\
            .option("header", "true")\
            .schema(schema)\
            .load("./Data/heart.csv")
        
        # Write it into MongoDB
        usersDataset.write\
            .format("com.mongodb.spark.sql.DefaultSource")\
            .option("uri","mongodb://127.0.0.1/mydb.heart")\
            .mode('append')\
            .save()

        # Read it back from MongoDB into a new Dataframe
        data = spark.read\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option("uri","mongodb://127.0.0.1/mydb.heart")\
        .load()

        # Cast columns to the appropriate data type
        data = data.select(
            col("age").cast(IntegerType()),
            col("sex").cast(IntegerType()),
            col("cp").cast(IntegerType()),
            col("trestbps").cast(IntegerType()),
            col("chol").cast(IntegerType()),
            col("fbs").cast(IntegerType()),
            col("restecg").cast(IntegerType()),
            col("thalach").cast(IntegerType()),
            col("exang").cast(IntegerType()),
            col("oldpeak").cast(DoubleType()),
            col("slope").cast(IntegerType()),
            col("ca").cast(IntegerType()),
            col("thal").cast(IntegerType()),
            col("heartdisease").cast(IntegerType())
        )

        #data.createOrReplaceTempView("users")
        
        #sqlDF = spark.sql("SELECT * FROM users WHERE age < 20")
        #sqlDF.show()

        # Load data as a dataframe
        #data = spark.read.option("header", "true").option("inferSchema", "true").csv(temp_file.name)

        # Assemble features into a vector column
        assembler = VectorAssembler().setInputCols(["age","sex","cp","trestbps","chol","fbs",\
                                                    "restecg","thalach","exang","oldpeak",\
                                                    "slope","ca","thal"]).\
                                                    setOutputCol("features")
                                                    
        df = assembler.transform(data).select("heartdisease", "features")
        
        # Split data into training and testing data
        trainTest = df.randomSplit([0.8, 0.2])
        trainingDF = trainTest[0]
        testDF = trainTest[1]

        # Train a Decision Tree model
        dtr = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("heartdisease")
        model = dtr.fit(trainingDF)

        # Save the trained model
        model.save("heart_model")

        # Predict on test data
        fullPredictions = model.transform(testDF).cache()
        predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
        labels = fullPredictions.select("heartdisease").rdd.map(lambda x: x[0])
        predictionAndLabel = predictions.zip(labels).collect()

        # Write the predictions to a CSV file
        with open('predictionsHeart.csv', mode='w') as predictions_file:
            predictions_writer = csv.writer(predictions_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            predictions_writer.writerow(['prediction', 'label'])
            for prediction in predictionAndLabel:
                predictions_writer.writerow(prediction)


        # Write predictions to in-memory CSV file
        csv_file = StringIO()
        predictions_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        predictions_writer.writerow(['prediction', 'label'])
        for prediction in predictionAndLabel:
            predictions_writer.writerow(prediction)

        # Convert CSV file to list of lists
        data = list(csv.reader(csv_file.getvalue().splitlines()))

        # Stop the SparkSession
        spark.stop()

        # Render the predictions in an HTML table
        #return render_template('predictions.html', data=data)

        
        """
        # Return the CSV file as a download
        response = make_response()
        response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
        with open(csv_file_path, mode='r') as csv_file:
            response.data = csv_file.read()

        # Stop the SparkSession
        spark.stop()

        return response
        """

        '''
        # Return the health recommendation
        return jsonify({'recommendation': recommendation})
        '''
        
        '''
        # Return predictions as a JSON object
        return jsonify(predictionAndLabel)
        '''
