# Databricks notebook source
import pyspark
from pyspark.sql import  SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as f
#from geopy.distance import geodesic
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

!pip install geopy
from geopy.distance import geodesic

# COMMAND ----------

spark = SparkSession.builder.appName(
    'Read CSV File into DataFrame').config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate()

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/2017_fordgobike_tripdataa_AM_PM.csv',sep=',' ,header=True)

# COMMAND ----------

df = df.withColumn('start_time', concat_ws(':', "start time hour", "start time minute", "start time seconds"))
df = df.withColumn('end_time', concat_ws(':', "end_time hour", "end_time minute", "end_time seconds"))
df = df.drop("start time hour","start time minute","start time seconds","end_time hour","end_time minute","end_time seconds")

# COMMAND ----------

df = df.withColumn("start_time_timestamp", to_timestamp("start_time", "H:mm:ss"))
df = df.withColumn("start_time_timestamp", date_format("start_time_timestamp", "H:mm:ss"))
df = df.withColumn("end_time_timestamp", to_timestamp("end_time", "H:mm:ss"))
df = df.withColumn("end_time_timestamp", date_format("end_time_timestamp", "H:mm:ss"))

# COMMAND ----------


@f.udf(returnType=FloatType())
def geodesic_udf(a, b):
    return geodesic(a, b).km


df = df.withColumn('Distance_kms', geodesic_udf(f.array("start_station_latitude", "start_station_longitude"),\
                                        f.array("end_station_latitude","end_station_longitude")))

# COMMAND ----------

# DBTITLE 1,1 . what is the ratio of payment using cc or app wallet
df_payment_count = df.groupby("pyment").count()
total_payment = df_payment_count.agg(sum("count")).collect()[0][0]
ratio = df_payment_count.withColumn("ratio",round((df_payment_count["count"]/total_payment)*100,2))
ratio.show()

# COMMAND ----------

df_ratio = ratio.toPandas()

# COMMAND ----------

plt.figure(figsize=(8, 6))

# Create the grouped bar plot
ax = sns.barplot(x='pyment', y='ratio', data=df_ratio,palette="Set3")
# Loop through bars and display numeric values
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='gray', xytext=(0, 5),
                textcoords='offset points')


# Add labels and title
plt.xlabel('Payments')
plt.ylabel('ratio')
plt.title('Payment ratio')
 

# Show the plot
plt.show()

# COMMAND ----------

# DBTITLE 1,2 . what is the preferred way to pay for customers and subscriber
ratio.orderBy("ratio",ascending=False).show()

# COMMAND ----------

# DBTITLE 1,3 . Analyze the relationship between trip duration and user type (subscriber vs. customer) to understand differences in usage patterns.
index = StringIndexer(inputCol="user_type", outputCol="user_type_encoded")
trips_encoded = index.fit(df).transform(df)

# COMMAND ----------

trip_user = trips_encoded.toPandas()

# COMMAND ----------

df_time_corr = trip_user[["user_type_encoded","Distance_kms"]].corr()
sns.heatmap(df_time_corr, annot=True, cmap="YlGnBu")
plt.show()
