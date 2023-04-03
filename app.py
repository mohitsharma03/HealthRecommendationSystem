from io import StringIO
import tempfile, csv, os
from flask import Flask, jsonify, render_template, request, Response, redirect, make_response, url_for
from pyspark.sql import SparkSession, Row, functions
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.sql.functions import col
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeRegressionModel
from pyspark.ml.feature import VectorAssembler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chose', methods=['POST'])
def chose():
        disease = request.form['disease']
        action = request.form['action']
        if disease=='diabetes':
            if action=='train':
                return redirect(url_for('trainDiabetes'))
            else:
                return redirect(url_for('singleUserDiabetes'))
        else:
            if action=='train':
                return redirect(url_for('trainHeart'))
            else:
                return redirect(url_for('singleUserHeart'))

@app.route('/trainDiabetes')
def trainDiabetes():
    return render_template('trainDiabetes.html')

@app.route('/singleUserDiabetes')
def singleUserDiabetes():
    return render_template('singleUserDiabetes.html')    

@app.route('/trainHeart')
def trainHeart():
    return render_template('trainHeart.html')

@app.route('/singleUserHeart')
def singleUserHeart():
    return render_template('singleUserHeart.html')  

@app.route('/predictPerUserDiabetes', methods=['POST'])
def predictPerUserDiabetes():
    # Create a SparkSession
    spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

    if request.method == 'POST':
        # Get the input values from the HTML form
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']
        skin_thickness = request.form['skin_thickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        # Create a new DataFrame object with a single row containing the input values
        schema = StructType([StructField("Pregnancies", IntegerType()), \
                             StructField("Glucose", DoubleType()), \
                             StructField("BloodPressure", DoubleType()), \
                             StructField("SkinThickness", DoubleType()), \
                             StructField("Insulin", DoubleType()), \
                             StructField("BMI", DoubleType()), \
                             StructField("DiabetesPedigreeFunction", DoubleType()), \
                             StructField("Age", IntegerType()), \
                             StructField("diabetes", IntegerType())])
        input_data = [(int(pregnancies), float(glucose), float(blood_pressure), \
                       float(skin_thickness), float(insulin), float(bmi), \
                       float(dpf), int(age), 0)]
        input_df = spark.createDataFrame(input_data, schema)

        # Assemble features into a vector column
        assembler = VectorAssembler().setInputCols(["Pregnancies","Glucose","BloodPressure",\
                                                    "SkinThickness","Insulin","BMI",\
                                                    "DiabetesPedigreeFunction","Age",]).\
                                                    setOutputCol("features")
                                                    
        testDF = assembler.transform(input_df).select("diabetes", "features")

        #Load the Decision Tree model
        model = DecisionTreeRegressionModel.load("diabetes_model")

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

@app.route('/predictDiabetes', methods=['POST'])
def predictDiabetes():
    # Create a SparkSession
    spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the file to a temporary file on disk
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        temp_file.close()

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
            .option("uri","mongodb://127.0.0.1/mydb.diabetes")\
            .mode('append')\
            .save()

        # Read it back from MongoDB into a new Dataframe
        data = spark.read\
        .format("com.mongodb.spark.sql.DefaultSource")\
        .option("uri","mongodb://127.0.0.1/mydb.diabetes")\
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
        
        # Save the trained model
        model.save("diabetes_model")
        #write.overwrite().save("diabetes_model")

        # Predict on test data
        fullPredictions = model.transform(testDF).cache()
        predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
        labels = fullPredictions.select("diabetes").rdd.map(lambda x: x[0])
        predictionAndLabel = predictions.zip(labels).collect()

        # Write the predictions to a CSV file
        with open('predictionsDiabetes.csv', mode='w') as predictions_file:
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
        return render_template('predictions.html', data=data)

@app.route('/predictPerUserHeart', methods=['POST'])
def predictPerUserHeart():
    # Create a SparkSession
    spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

    if request.method == 'POST':
         # Get the form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Create a new DataFrame object with a single row containing the input values
        schema = StructType([StructField("age", IntegerType()), \
                             StructField("sex", IntegerType()), \
                             StructField("cp", IntegerType()), \
                             StructField("trestbps", IntegerType()), \
                             StructField("chol", IntegerType()), \
                             StructField("fbs", IntegerType()), \
                             StructField("restecg", IntegerType()), \
                             StructField("thalach", IntegerType()), \
                             StructField("exang", IntegerType()), \
                             StructField("oldpeak", DoubleType()), \
                             StructField("slope", IntegerType()), \
                             StructField("ca", IntegerType()), \
                             StructField("thal", IntegerType()), \
                             StructField("heartdisease", IntegerType())])
        
        input_data = [(age, sex, cp, trestbps, chol, fbs, restecg, thalach, \
                        exang, oldpeak, slope, ca, thal, 0)]
        input_df = spark.createDataFrame(input_data, schema)

        # Assemble features into a vector column
        assembler = VectorAssembler().setInputCols(["age","sex","cp","trestbps","chol","fbs",\
                                                    "restecg","thalach","exang","oldpeak",\
                                                    "slope","ca","thal"]).\
                                                    setOutputCol("features")
                                                    
        testDF = assembler.transform(input_df).select("heartdisease", "features")

        # Load the Decision Tree model
        model = DecisionTreeRegressionModel.load("heart_model")

        # Predict on test data
        fullPredictions = model.transform(testDF).cache()
        predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
        labels = fullPredictions.select("heartdisease").rdd.map(lambda x: x[0])
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


@app.route('/predictHeart', methods=['POST'])
def predictHeart():
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

        #data.createOrReplaceTempView("users")
        
        #sqlDF = spark.sql("SELECT * FROM users WHERE age < 20")
        #sqlDF.show()

        # Load data as a dataframe
        #data = spark.read.option("header", "true").option("inferSchema", "true").csv(temp_file.name)

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
        #write.overwrite().save("heart_model")

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
        return render_template('predictions.html', data=data)

        
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

if __name__ == '__main__':
    app.run(debug=True)
