Below are the steps to setup the environment:
**Note:** the setup is done in ubuntu 22.04 LTS:

Setup spark on ubuntu:

Check python version(3.5 is needed for spark 2.1):
python3

To update python version:
sudo apt-get update && sudo apt-get upgrade 
sudo apt-get install python3.5

If pip is not installed:
sudo apt install python3-pip

To install jupyter notebook:
sudo -H pip install jupyter

Open jupyter notebook: run below command and if opening jupyter notebook for first time then copy paste the url u get from terminal in ur browser:
jupyter notebook

Before installing spark, u need to insatll java & scala:
1)Install java:
If java installed then check version: java -version
Else install it:
sudo apt-get update
sudo apt-get install default-jre
2)Install scala:
If java installed then check version: scala -version
Else install it:
sudo apt-get update
sudo apt-get install scala
3)install library py4j wch will connect java & scala with spark:
pip3 install py4j

Install spark:
1)Ubuntu 22.04: spark.apache.org->Downloads->select spark release 3.3.2, select package type(pre-built hadoop 3.3 & later), select download type(direct download)->then click download spark
Note: my python version was: 3.10.6, java: 11.0.18, scala: 2.11.12

For Ubuntu 16:
spark.apache.org->Downloads->select spark release, select package type(pre-built hadoop 2.7 & later), select download type(direct downlaod)->then click download spark
OR, to download spark 2.1.0, download from here: https://archive.apache.org/dist/spark/spark-2.1.0/
2)Move the tar file to home folder
3)Terminal: 
sudo tar -zxvf spark<tab>
export SPARK_HOME=’home/ubuntu/spark-3.3.2-bin-hadoop3’

export PATH=$SPARK_HOME:$PATH
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=’jupyter’
export PYSPARK_DRIVER_PYTHON_OPTS=’notebook’
export PYSPARK_PYTHON=python3
sudo chmod 777 spark<tab>
cd spark<tab>
cd python
python3
>>import pyspark
>>quit()

cd ..
sudo chmod 777 python
cd python
sudo chmod 777 pyspark

jupyter notebook
In browser, open python3 in jupyter notebook, run below command and shift+enter:
import pyspark

How to setup such that we can run ‘import pyspark’ from anywhere instead of going into spark folder everytime:
Move to home directory: cd
pip3 install findspark
To get path-of-spark-folder: cd to the folder & then ‘pwd’ & then cd ..
python3
>>import findspark
>>findspark.init(‘<path-of-spark-folder>’)
>>import pyspark

To do the same in jupyter notebook:
Just open jupyter notebook from home
Go to browser
Open python3 and execute:
import findspark
findspark.init(‘<path-of-spark-folder>’)
import pyspark

Jupyter Notebook Usage:

<variable-name>.<tab>: u can see all methods available
U can use Help tab also

Running spark from python file:
1)Set Environment Variables: If not set then u will get issue for below command: bash: spark-submit: command not found….Set the environment variables for Spark in your terminal by adding the following lines to your ~/.bashrc file
i) Open file: nano ~/.bashrc
ii) Add lines: 
export SPARK_HOME=/home/mohit/spark-3.3.2-bin-hadoop3
OR, export SPARK_HOME=/opt/spark-3.3.2-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH
iv) Source the .bashrc file to apply the changes: source ~/.bashrc

2)Run the Python code: Once you have set up Spark, you can run the Python code by navigating to the directory where the code is saved and running the following command:
spark-submit health_recommender.py


—--------------------------------------------------------
Mongodb setup for spark in Ubuntu 22.04 LTS:

mongod –version
mongodb install un ubuntu: 
https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/?_ga=2.20164474.318501314.1680199481-1609137488.1680199480

service mongod status
sudo service mongod start
service mongod status

sudo systemctl enable mongod.service: will start mongodb server whenever the system boots up

Dint wrk: /usr/bin/mongo: to enter into mongo shell 
mongosh:  to enter into mongo shell
>>show dbs
>>use mydb
>>show collections
>> db.createCollection("mycollection")
>>db.mycollection.count()
>>db.mycollection.count()
>>db.mycollection.find({})
>>db.mycollection.drop()
>>db.dropDatabase()
>>quit()


Error for mongodb-connector for spark:
py4j.protocol.Py4JJavaError: An error occurred while calling o50.save.
: java.lang.ClassNotFoundException: 
Failed to find data source: com.mongodb.spark.sql.DefaultSource. Please find packages at
https://spark.apache.org/third-party-projects.html

Solution for Mongo-db connector issue for spark:
We need to put mongo-spark-connector(chose it according to the scala version present in ur m/c) and mongo-java-driver jar in spark->jars folder: taken from ASHWINI S soln given here https://stackoverflow.com/questions/50963444/failed-to-find-data-source-com-mongodb-spark-sql-defaultsource
Jars downloaded frm here:
https://mvnrepository.com/artifact/org.mongodb.spark/mongo-spark-connector_2.12/2.4.4
https://mvnrepository.com/artifact/org.mongodb/mongo-java-driver/3.10.2


Using mongodb in spark:
pip install pyspark
https://github.com/datyrlab/apache-spark/blob/master/04-mongodb-connector.py

---------------------------------------------------------------------------

