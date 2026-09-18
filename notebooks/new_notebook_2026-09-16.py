# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW parsed_wsp AS
# MAGIC SELECT ai_parse_document(content) AS parsed
# MAGIC FROM READ_FILES('/Volumes/workspace/default/raw/WSP_Intersecting_Streams_Unregulated_2024.pdf', format => 'binaryFile')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW wsp_chunks AS
# MAGIC SELECT
# MAGIC   chunk.value:chunk_id::STRING AS chunk_id,
# MAGIC   chunk.value:chunk_to_embed::STRING AS text,
# MAGIC   chunk.value:pages AS pages
# MAGIC FROM (SELECT ai_prep_search(parsed) AS result FROM parsed_wsp),
# MAGIC LATERAL variant_explode(result:document.contents) AS chunk

# COMMAND ----------

df = spark.sql("""
  SELECT chunk_id, text,
         ai_query('databricks-gte-large-en', text) AS embedding
  FROM wsp_chunks
""")
display(df)