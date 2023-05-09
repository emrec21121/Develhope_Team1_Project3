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

# COMMAND ----------

!pip install geopy
from geopy.distance import geodesic

# COMMAND ----------

spark = SparkSession.builder.appName(
    'Read CSV File into DataFrame').config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate()

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/2017_fordgobike_tripdataa_AM_PM.csv',sep=',' ,header=True)

# COMMAND ----------



# COMMAND ----------

display(df)

# COMMAND ----------

df = df.withColumn('start_time', concat_ws(':', "start time hour", "start time minute", "start time seconds"))
df = df.withColumn('end_time', concat_ws(':', "end_time hour", "end_time minute", "end_time seconds"))
df = df.drop("start time hour","start time minute","start time seconds","end_time hour","end_time minute","end_time seconds")

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.withColumn("start_time_timestamp", to_timestamp("start_time", "H:mm:ss"))
df = df.withColumn("start_time_timestamp", date_format("start_time_timestamp", "H:mm:ss"))
df = df.withColumn("end_time_timestamp", to_timestamp("end_time", "H:mm:ss"))
df = df.withColumn("end_time_timestamp", date_format("end_time_timestamp", "H:mm:ss"))



# COMMAND ----------

df = df.withColumn("start_time", when(df.start_time_timestamp < df.end_time_timestamp, df.start_time_timestamp).otherwise(df.end_time_timestamp))
df = df.withColumn("end_time", when(df.start_time_timestamp < df.end_time_timestamp, df.end_time_timestamp).otherwise(df.start_time_timestamp))
df = df.drop("start_time_timestamp", "end_time_timestamp")
df.show()

# COMMAND ----------

et_unix = unix_timestamp(to_timestamp(df.end_time,"H:mm:ss"),"H:mm:ss")
st_unix = unix_timestamp(to_timestamp(df.start_time,"H:mm:ss"),"H:mm:ss")
df = df.withColumn("duration_seconds", et_unix - st_unix )

# COMMAND ----------

df = df.withColumn("cost_per_ride", (df.duration_seconds/60)*0.35)

# COMMAND ----------

User_type_total =df.groupby("user_type").sum("cost_per_ride").withColumnRenamed("sum(cost_per_ride)","Total_users_cost") \
User_type_total.withColumn("Total_users_cost",round(User_type_total["Total_users_cost"]))
                                                             


# COMMAND ----------

display(User_type_total)

# COMMAND ----------

df_day = df.select("start_station_id","start time hour","_c4")
display(df_day.distinct())

# COMMAND ----------

df_day = df_day.withColumn("time_interval", \
                  when((df_day["start time hour"] >= 4) & (df_day["start time hour"] <= 10) & (df_day["_c4"] == "AM"),"Morning")
                  .when((df_day["start time hour"] >= 11) & (df_day["start time hour"] <= 12) & (df_day["_c4"] == "AM"),"Afternoon")
                  .when((df_day["start time hour"] >= 1) & (df_day["start time hour"] <= 3) & (df_day["_c4"] == "AM"),"Night")
                  .when((df_day["start time hour"] >= 1) & (df_day["start time hour"] <= 4) & (df_day["_c4"] == "PM"),"Afternoon")
                  .when((df_day["start time hour"] >= 5) & (df_day["start time hour"] <=10 ) & (df_day["_c4"] == "PM"),"Everning")
                  .when((df_day["start time hour"] >= 11) & (df_day["start time hour"] <=12 ) & (df_day["_c4"] == "PM"),"Night"))
                  

# COMMAND ----------

display(df_day_grouped)


# COMMAND ----------

df_day_grouped = df_day.groupby("time_interval").count().withColumnRenamed("count","Density")

# COMMAND ----------

pydf_day = df_day_grouped.toPandas()

# COMMAND ----------

plt.figure(figsize=(8, 6))

# Create the grouped bar plot
ax = sns.barplot(x='time_interval', y='Density', data=pydf_day,alpha=0.5,palette="Set2")
# Loop through bars and display numeric values
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='gray', xytext=(0, 5),
                textcoords='offset points')


# Add labels and title
plt.xlabel('Day_Intervals')
plt.ylabel('Density')
plt.title('Application usage in a day')
 

# Show the plot
plt.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

df.select("date_time").distinct().show()

# COMMAND ----------

pydf_day

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_time_relation = df.select(hour("start_time").alias("start_time"),"start_station_name","start_station_id")
df_time_relation.show()

 

# COMMAND ----------

df_time_relation = df_time_relation.withColumn("start_station_id", df_time_relation["start_station_id"].cast("integer"))

df_time_relation.printSchema()

# COMMAND ----------

df_time = df_time_relation.select("*")

# COMMAND ----------

df_time = df_time.toPandas()

# COMMAND ----------

df_time_corr = df_time[["start_time","start_station_id"]].corr()
df_time_corr


# COMMAND ----------

sns.heatmap(df_time_corr, annot=True, cmap="YlGnBu")


plt.show()
