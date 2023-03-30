from io import StringIO
import tempfile
import csv
from flask import Flask, jsonify, render_template, request, Response, redirect, make_response
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.functions import col

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Create a SparkSession
    spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the file to a temporary file on disk
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        temp_file.close()

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
            StructField("Pregnancies", IntegerType(), True),
            StructField("Glucose", DoubleType(), True),
            StructField("BloodPressure", DoubleType(), True),
            StructField("SkinThickness", DoubleType(), True),
            StructField("Insulin", DoubleType(), True),
            StructField("BMI", DoubleType(), True),
            StructField("DiabetesPedigreeFunction", DoubleType(), True),
            StructField("Age", IntegerType(), True),
            StructField("diabetes", IntegerType(), True)
        ])

        # Read the CSV file using the specified schema
        usersDataset = spark.read.format("csv")\
            .option("header", "true")\
            .schema(schema)\
            .load(temp_file.name)
        
        # Write it into MongoDB
        usersDataset.write\
            .format("com.mongodb.spark.sql.DefaultSource")\
            .option("uri","mongodb://127.0.0.1/mydb.users")\
            .mode('append')\
            .save()

        # Read it back from MongoDB into a new Dataframe
        data = spark.read\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option("uri","mongodb://127.0.0.1/mydb.users")\
        .load()

        # Cast columns to the appropriate data type
        data = data.select(
            col("Pregnancies").cast(IntegerType()),
            col("Glucose").cast(DoubleType()),
            col("BloodPressure").cast(DoubleType()),
            col("SkinThickness").cast(DoubleType()),
            col("Insulin").cast(DoubleType()),
            col("BMI").cast(DoubleType()),
            col("DiabetesPedigreeFunction").cast(DoubleType()),
            col("Age").cast(IntegerType()),
            col("diabetes").cast(IntegerType())
        )

        #data.createOrReplaceTempView("users")
        
        #sqlDF = spark.sql("SELECT * FROM users WHERE age < 20")
        #sqlDF.show()

        # Load data as a dataframe
        #data = spark.read.option("header", "true").option("inferSchema", "true").csv(temp_file.name)

        # Assemble features into a vector column
        assembler = VectorAssembler().setInputCols(["Pregnancies","Glucose","BloodPressure",\
                                                    "SkinThickness","Insulin","BMI",\
                                                    "DiabetesPedigreeFunction","Age",]).\
                                                    setOutputCol("features")
                                                    
        df = assembler.transform(data).select("diabetes", "features")
        
        # Split data into training and testing data
        trainTest = df.randomSplit([0.8, 0.2])
        trainingDF = trainTest[0]
        testDF = trainTest[1]

        # Train a Decision Tree model
        dtr = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("diabetes")
        model = dtr.fit(trainingDF)

        # Predict on test data
        fullPredictions = model.transform(testDF).cache()
        predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
        labels = fullPredictions.select("diabetes").rdd.map(lambda x: x[0])
        predictionAndLabel = predictions.zip(labels).collect()

        
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
        return render_template('predictions.html', data=data)
        
if __name__ == '__main__':
    app.run(debug=True)
