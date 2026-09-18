# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # NSW Water Sharing Plans — AI Experiment
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC
# MAGIC 1. **Parse** multiple NSW Water Sharing Plan PDFs using `ai_parse_document`
# MAGIC 2. **Chunk** each plan into searchable text segments using `ai_prep_search`
# MAGIC 3. **Store** all chunks in a Delta table with plan-level metadata
# MAGIC 4. **Index** the corpus with a Vector Search endpoint for semantic search
# MAGIC 5. **Compare** provisions across plans using semantic similarity and AI Q&A
# MAGIC
# MAGIC **Source PDFs** (in `/Volumes/workspace/default/raw/`):
# MAGIC - `WSP_Barwon-Darling_Unregulated_2026.pdf`
# MAGIC - `WSP_Intersecting_Streams_Unregulated_2024.pdf`
# MAGIC
# MAGIC Add more PDFs to the volume and re-run the pipeline to include them automatically.

# COMMAND ----------

# DBTITLE 1,Step 1 — Parse all PDFs
# MAGIC %sql
# MAGIC -- Parse every PDF in the volume directory in one shot
# MAGIC -- READ_FILES with format => 'binaryFile' returns raw BINARY content per file
# MAGIC -- ai_parse_document extracts structured elements (text, tables, headers, figures)
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.default.wsp_parsed_docs AS
# MAGIC SELECT
# MAGIC   _metadata.file_name                                       AS file_name,
# MAGIC   regexp_replace(_metadata.file_name, '\\.pdf$', '')         AS plan_name,
# MAGIC   content,
# MAGIC   ai_parse_document(content, MAP('version', '2.0'))          AS parsed_content
# MAGIC FROM READ_FILES('/Volumes/workspace/default/raw/', format => 'binaryFile')
# MAGIC WHERE path LIKE '%.pdf'
# MAGIC   AND _metadata.file_name LIKE 'WSP_%';
# MAGIC
# MAGIC -- Quick check: how many plans, and did parsing succeed?
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   CASE WHEN is_variant_null(parsed_content:error_status) THEN 'OK' ELSE 'ERROR' END AS parse_status,
# MAGIC   size(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS element_count
# MAGIC FROM workspace.default.wsp_parsed_docs
# MAGIC ORDER BY plan_name;

# COMMAND ----------

# DBTITLE 1,Step 2 — Chunk all plans
# MAGIC %sql
# MAGIC -- Chunk each parsed plan into semantically meaningful text segments
# MAGIC -- ai_prep_search takes the parsed VARIANT and returns an array of chunk strings
# MAGIC -- We explode the array and attach plan-level metadata + a deterministic chunk_id
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.default.wsp_chunks
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC AS
# MAGIC SELECT
# MAGIC   md5(concat(plan_name, '_', cast(chunk:chunk_position AS STRING))) AS chunk_id,
# MAGIC   plan_name,
# MAGIC   file_name,
# MAGIC   chunk:chunk_position::INT  AS chunk_index,
# MAGIC   chunk:chunk_to_embed::STRING AS chunk_text
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     plan_name,
# MAGIC     file_name,
# MAGIC     explode(try_cast(ai_prep_search(parsed_content):document:contents AS ARRAY<VARIANT>)) AS chunk
# MAGIC   FROM workspace.default.wsp_parsed_docs
# MAGIC   WHERE is_variant_null(parsed_content:error_status)
# MAGIC );
# MAGIC
# MAGIC -- Preview: how many chunks per plan, and what do they look like?
# MAGIC SELECT plan_name, count(*) AS chunk_count, avg(length(chunk_text)) AS avg_chunk_len
# MAGIC FROM workspace.default.wsp_chunks
# MAGIC GROUP BY plan_name
# MAGIC ORDER BY plan_name;
# MAGIC
# MAGIC SELECT plan_name, chunk_index, substring(chunk_text, 1, 200) AS chunk_preview
# MAGIC FROM workspace.default.wsp_chunks
# MAGIC ORDER BY plan_name, chunk_index
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Step 3 — Prepare table for Vector Search
# MAGIC %sql
# MAGIC -- Vector Search Delta Sync indexes require:
# MAGIC --   1. Change Data Feed (CDF) enabled on the source table
# MAGIC --   2. A PRIMARY KEY constraint on the column used as primary_key
# MAGIC --   3. The primary key column must be NOT NULL
# MAGIC -- CDF is already enabled in the CREATE TABLE (cell above).
# MAGIC -- Here we set NOT NULL and add the PK constraint.
# MAGIC
# MAGIC ALTER TABLE workspace.default.wsp_chunks
# MAGIC   ALTER COLUMN chunk_id SET NOT NULL;
# MAGIC
# MAGIC ALTER TABLE workspace.default.wsp_chunks
# MAGIC   ADD CONSTRAINT wsp_chunks_pk PRIMARY KEY (chunk_id);

# COMMAND ----------

# DBTITLE 1,Step 4 — Create Vector Search endpoint & index
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()

endpoint_name = "wsp-vs-endpoint"
index_name = "workspace.default.wsp_chunks_index"

# --- Create the Vector Search endpoint (if it doesn't exist yet) ---
try:
    w.vector_search_endpoints.get_endpoint(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists.")
except NotFound:
    print(f"Creating endpoint '{endpoint_name}'…")
    w.vector_search_endpoints.create_endpoint(
        name=endpoint_name,
        endpoint_type=EndpointType.STANDARD,
    )
    # Wait for it to come online
    import time
    for _ in range(60):
        ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
        state = str(ep.endpoint_status.state)
        print(f"  Endpoint state: {state}")
        if "ONLINE" in state:
            break
        time.sleep(10)

# --- Create the Delta Sync index (managed embeddings) ---
# The index will auto-compute embeddings for chunk_text using databricks-gte-large-en
# and sync from the Delta table automatically.
try:
    w.vector_search_indexes.get_index(index_name=index_name)
    print(f"Index '{index_name}' already exists.")
except NotFound:
    print(f"Creating index '{index_name}'…")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint_name,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table="workspace.default.wsp_chunks",
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="chunk_text",
                    embedding_model_endpoint_name="databricks-gte-large-en",
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
        ),
    )
    print("Index creation started. Waiting for it to be ready…")
    import time
    for _ in range(60):
        idx = w.vector_search_indexes.get_index(index_name=index_name)
        ready = idx.status.ready
        print(f"  Index ready: {ready}")
        if ready:
            break
        time.sleep(10)
    if not ready:
        raise RuntimeError("Index did not become ready in time. Check the Vector Search UI.")

# --- Trigger a sync so embeddings are computed now ---
print("Triggering index sync…")
try:
    w.vector_search_indexes.sync_index(index_name=index_name)
    print("Sync triggered. Check the Vector Search UI for progress.")
except Exception as e:
    print(f"Index not ready for sync yet: {e}")
    print("The index may take 10-20 minutes to provision on a new endpoint.")
    print("Re-run this cell later, or check: https://your-workbench/explore/data/workspace/default/wsp_chunks_index")

# COMMAND ----------

# DBTITLE 1,Step 5 — Semantic search across all plans
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound

w = WorkspaceClient()

query = "What are the daily access rules and extraction limits?"
index_name = "workspace.default.wsp_chunks_index"

print(f"Query: {query}\n")

try:
    # Try Vector Search index first (once it's ready)
    results = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["plan_name", "file_name", "chunk_index", "chunk_text"],
        query_text=query,
        num_results=10,
    )
    for i, row in enumerate(results.result.data_array, 1):
        plan_name = row[0]
        chunk_text = row[3]
        score = row[-1]
        print(f"--- Result {i} | Score: {score:.4f} | Plan: {plan_name} ---")
        print(chunk_text[:300])
        print()
    plans_in_results = set(row[0] for row in results.result.data_array)
    print(f"Plans found in top results: {plans_in_results}")
    print("\n(Results via Vector Search index)")
except Exception as e:
    print(f"Vector Search not available yet ({type(e).__name__}). Falling back to ai_similarity...\n")
    # Fallback: use ai_similarity via SQL for retrieval
    import pandas as pd
    df = spark.sql(f"""
        SELECT plan_name, chunk_index, chunk_text,
               ai_similarity(chunk_text, '{query}') AS score
        FROM workspace.default.wsp_chunks
        ORDER BY score DESC
        LIMIT 10
    """).toPandas()
    for i, row in df.iterrows():
        print(f"--- Result {i+1} | Score: {row['score']:.4f} | Plan: {row['plan_name']} ---")
        print(row['chunk_text'][:300])
        print()
    plans_in_results = set(df['plan_name'])
    print(f"Plans found in top results: {plans_in_results}")
    print("\n(Results via ai_similarity fallback)")

# COMMAND ----------

# DBTITLE 1,Step 6 — Find similar provisions across plans
# MAGIC %sql
# MAGIC -- Cross-plan comparison: find provisions that are semantically similar
# MAGIC -- between the two water sharing plans using ai_similarity().
# MAGIC -- This helps identify overlapping rules, shared clauses, or divergent wording.
# MAGIC --
# MAGIC -- NOTE: CROSS JOIN cost scales as O(n²). For large corpora, use the Vector
# MAGIC -- Search index to find nearest neighbours per chunk instead. This is fine
# MAGIC -- for two plans with a few hundred chunks each.
# MAGIC
# MAGIC SELECT
# MAGIC   a.plan_name  AS plan_a,
# MAGIC   b.plan_name  AS plan_b,
# MAGIC   a.chunk_index AS chunk_a_idx,
# MAGIC   b.chunk_index AS chunk_b_idx,
# MAGIC   substring(a.chunk_text, 1, 150)  AS chunk_a,
# MAGIC   substring(b.chunk_text, 1, 150)  AS chunk_b,
# MAGIC   ai_similarity(a.chunk_text, b.chunk_text) AS similarity
# MAGIC FROM workspace.default.wsp_chunks a
# MAGIC CROSS JOIN workspace.default.wsp_chunks b
# MAGIC WHERE a.plan_name < b.plan_name
# MAGIC   AND ai_similarity(a.chunk_text, b.chunk_text) > 0.70
# MAGIC ORDER BY similarity DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Step 7 — AI Q&A across the corpus
# MAGIC %sql
# MAGIC -- Use a foundation model to answer questions about the plans.
# MAGIC -- We retrieve the most relevant chunks using ai_similarity (no Vector Search needed),
# MAGIC -- then ask an LLM to synthesise an answer grounded in those chunks.
# MAGIC -- Replace the question text with any query about the water sharing plans.
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW retrieved_chunks AS
# MAGIC SELECT plan_name, chunk_index, chunk_text, score
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     plan_name,
# MAGIC     chunk_index,
# MAGIC     chunk_text,
# MAGIC     ai_similarity(chunk_text, 'What are the rules for managing floods in these water sharing plans?') AS score
# MAGIC   FROM workspace.default.wsp_chunks
# MAGIC )
# MAGIC ORDER BY score DESC
# MAGIC LIMIT 10;
# MAGIC
# MAGIC SELECT plan_name, chunk_index, score, substring(chunk_text, 1, 200) AS chunk_preview
# MAGIC FROM retrieved_chunks
# MAGIC ORDER BY score DESC;
# MAGIC
# MAGIC -- Then, pass the retrieved context to an LLM for a synthesised answer:
# MAGIC
# MAGIC SELECT ai_query(
# MAGIC   'databricks-llama-4-maverick',
# MAGIC   concat(
# MAGIC     'You are analysing NSW Water Sharing Plans. Based on the following excerpts, ',
# MAGIC     'answer the question: "What are the rules for managing floods in these water sharing plans?"\n\n',
# MAGIC     'If the excerpts do not contain enough information, say so. ',
# MAGIC     'Cite the plan name for each point.\n\n',
# MAGIC     array_join(collect_list(concat('Plan: ', plan_name, '\n', chunk_text)), '\n---\n')
# MAGIC   )
# MAGIC ) AS answer
# MAGIC FROM retrieved_chunks;

# COMMAND ----------

# DBTITLE 1,Step 8 — Per-plan summaries
# MAGIC %sql
# MAGIC -- Generate a concise AI summary of each water sharing plan independently.
# MAGIC -- This is useful for quickly understanding what each plan covers.
# MAGIC
# MAGIC WITH plan_text AS (
# MAGIC   SELECT
# MAGIC     plan_name,
# MAGIC     array_join(collect_list(chunk_text), '\n\n') AS full_text
# MAGIC   FROM workspace.default.wsp_chunks
# MAGIC   GROUP BY plan_name
# MAGIC )
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   ai_query(
# MAGIC     'databricks-llama-4-maverick',
# MAGIC     concat(
# MAGIC       'Summarise the key features of this NSW Water Sharing Plan in 5 bullet points. ',
# MAGIC       'Focus on water access rules, environmental water, and compliance.\n\n',
# MAGIC       substring(full_text, 1, 12000)
# MAGIC     )
# MAGIC   ) AS summary
# MAGIC FROM plan_text
# MAGIC ORDER BY plan_name;

# COMMAND ----------

# DBTITLE 1,Step 9 — Monitor AI function token/DBU consumption
# MAGIC %sql
# MAGIC -- AI functions (ai_parse_document, ai_query, ai_similarity, ai_prep_search) are billed
# MAGIC -- as SERVERLESS_REAL_TIME_INFERENCE DBUs. This query shows your daily AI function costs.
# MAGIC --
# MAGIC -- Note: billing data typically has a 12-24 hour delay before it appears here.
# MAGIC -- For per-token granularity, you would need to route calls through an AI Gateway endpoint
# MAGIC -- (which populates system.ai_gateway.usage with input_tokens, output_tokens, etc.).
# MAGIC
# MAGIC SELECT
# MAGIC   u.usage_date,
# MAGIC   u.sku_name,
# MAGIC   u.usage_type,
# MAGIC   sum(u.usage_quantity) AS total_dbus,
# MAGIC   round(sum(u.usage_quantity * p.pricing.effective_list.default), 4) AS estimated_cost_usd
# MAGIC FROM system.billing.usage u
# MAGIC LEFT JOIN system.billing.list_prices p
# MAGIC   ON u.sku_name = p.sku_name AND u.cloud = p.cloud
# MAGIC WHERE u.usage_date >= current_date() - 7
# MAGIC   AND u.sku_name LIKE '%REAL_TIME_INFERENCE%'
# MAGIC GROUP BY u.usage_date, u.sku_name, u.usage_type
# MAGIC ORDER BY u.usage_date DESC, u.usage_type;

# COMMAND ----------

# DBTITLE 1,Step 10 — Create Knowledge Assistant (chat UI)
# --- Setting up a chat UI for end users ---
#
# Knowledge Assistant (Agent Bricks) is not available in this workspace.
# Two alternatives work with the existing Vector Search index:
#
# OPTION 1 — AI Playground (fastest, for testing)
#   1. Open AI Playground from the workspace sidebar (AI > AI Playground)
#   2. Select a foundation model (e.g. databricks-llama-4-maverick)
#   3. Toggle on "Retrieve" / RAG mode
#   4. Select the Vector Search index: workspace.default.wsp_chunks_index
#   5. Set the text column to chunk_text
#   6. Ask questions — the playground retrieves relevant chunks and grounds the answer
#
# OPTION 2 — Databricks App (production-ready, shareable UI)
#   Run the next cell to see a Streamlit app you can deploy as a Databricks App.
#   The app queries the Vector Search index and synthesises answers with an LLM.

print("See comments above for AI Playground instructions.")
print("Run the next cell to generate a Streamlit chat app for a shareable UI.")

# COMMAND ----------

# DBTITLE 1,Step 11 — Streamlit chat app (for Databricks App deployment)
# This cell generates app.py — a Streamlit chat app you can deploy as a Databricks App.
# To deploy:
#   1. Create a Databricks App (Apps > Create App > streamlit)
#   2. Copy this code into the app's app.py file
#   3. Add 'databricks-sdk' and 'streamlit' to requirements.txt
#   4. Deploy and share the URL with users
#
# The app queries the Vector Search index for relevant chunks,
# then asks an LLM to synthesise a cited answer.

app_code = '''"""NSW Water Sharing Plans Q&A — Streamlit Chat App"""
import streamlit as st
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
INDEX_NAME = "workspace.default.wsp_chunks_index"
MODEL = "databricks-llama-4-maverick"

st.title("NSW Water Sharing Plans Q&A")
st.caption("Ask questions about the Barwon-Darling and Intersecting Streams water sharing plans.")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Source chunks"):
                for src in msg["sources"]:
                    st.write(f"**{src['plan']}** (chunk {src['chunk_idx']}, score {src['score']:.3f})")
                    st.text(src["text"][:500] + "...")

# Chat input
if question := st.chat_input("Ask about water sharing plans..."):
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching plans..."):
            # 1. Retrieve relevant chunks via Vector Search
            results = w.vector_search_indexes.query_index(
                index_name=INDEX_NAME,
                columns=["plan_name", "chunk_index", "chunk_text"],
                query_text=question,
                num_results=5,
            )
            chunks = []
            for row in results.result.data_array:
                chunks.append({
                    "plan": row[0],
                    "chunk_idx": row[1],
                    "text": row[2],
                    "score": row[-1],
                })

            if not chunks:
                answer = "No relevant passages found in the water sharing plans."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
                st.stop()

            # 2. Build context for the LLM
            context = "\\n---\\n".join(
                f"Plan: {c['plan']}\\n{c['text']}" for c in chunks
            )

            # 3. Ask the LLM (via SQL ai_query for simplicity)
            import json
            prompt = (
                f"You are analysing NSW Water Sharing Plans. "
                f"Answer this question based on the excerpts below: \\n\\n"
                f"Question: {question}\\n\\n"
                f"Excerpts:\\n{context}\\n\\n"
                f"Cite the plan name for each point. If the excerpts don't answer the question, say so."
            )
            answer_df = spark.sql(f"""
                SELECT ai_query('{MODEL}', '{prompt.replace("'", "''")}') AS answer
            """)
            answer = answer_df.collect()[0]["answer"]

            st.markdown(answer)
            with st.expander("Source chunks"):
                for src in chunks:
                    st.write(f"**{src['plan']}** (chunk {src['chunk_idx']}, score {src['score']:.3f})")
                    st.text(src["text"][:500] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks,
            })
'''

print(app_code)
print("\n--- To deploy as a Databricks App ---")
print("1. Go to Apps in the workspace sidebar > Create App > streamlit")
print("2. Replace app.py with the code above")
print("3. Add 'databricks-sdk' and 'streamlit' to requirements.txt")
print("4. Deploy and share the URL with users")

# COMMAND ----------

# DBTITLE 1,Step 12 — Extract rules from tables (deterministic)
# MAGIC %sql
# MAGIC -- Extract structured rules from tables in the parsed documents
# MAGIC -- Filter parsed_content:document:elements for type='table' and extract numeric data
# MAGIC -- (flow class thresholds, extraction limits, access conditions)
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW wsp_table_rules AS
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   'table' AS extraction_source,
# MAGIC   element:element_id::STRING AS element_id,
# MAGIC   element:type::STRING AS element_type,
# MAGIC   element:text::STRING AS table_text,
# MAGIC   -- Parse table structure: detect numeric flow thresholds, extraction limits
# MAGIC   -- This is a simplified example - real tables need regex/parsing of table_text
# MAGIC   CASE
# MAGIC     WHEN lower(element:text::STRING) LIKE '%flow class%' OR lower(element:text::STRING) LIKE '%threshold%' THEN 'flow_class_threshold'
# MAGIC     WHEN lower(element:text::STRING) LIKE '%extraction%limit%' OR lower(element:text::STRING) LIKE '%annual%limit%' THEN 'extraction_limit'
# MAGIC     WHEN lower(element:text::STRING) LIKE '%cease%pump%' OR lower(element:text::STRING) LIKE '%cease%take%' THEN 'cease_to_pump'
# MAGIC     WHEN lower(element:text::STRING) LIKE '%access%' THEN 'access_condition'
# MAGIC     ELSE 'other'
# MAGIC   END AS rule_type,
# MAGIC   NULL AS water_source,      -- Would extract from table headers/context
# MAGIC   NULL AS condition,          -- Would parse from table rows
# MAGIC   NULL AS value,              -- Would extract numeric values
# MAGIC   NULL AS unit,               -- Would extract units (ML, GL, ML/day)
# MAGIC   NULL AS applies_to,         -- Would extract license category
# MAGIC   NULL AS section_reference,  -- Could link to section from element metadata
# MAGIC   ARRAY() AS citation_ids,    -- Tables have element_id as citation
# MAGIC   NULL AS confidence_score    -- Deterministic extraction has no confidence score
# MAGIC FROM workspace.default.wsp_parsed_docs
# MAGIC LATERAL VIEW explode(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS element
# MAGIC WHERE element:type::STRING = 'table'
# MAGIC   AND is_variant_null(parsed_content:error_status);
# MAGIC
# MAGIC -- Preview: how many table elements per plan?
# MAGIC SELECT plan_name, count(*) AS table_count
# MAGIC FROM wsp_table_rules
# MAGIC GROUP BY plan_name
# MAGIC ORDER BY plan_name;
# MAGIC
# MAGIC SELECT * FROM wsp_table_rules LIMIT 5;

# COMMAND ----------

# DBTITLE 1,Step 13 — Extract rules from text chunks (ai_extract v2.1)
# MAGIC %sql
# MAGIC -- Extract structured rules from text chunks using ai_extract v2.1
# MAGIC -- with citations and confidence scores enabled.
# MAGIC -- Schema uses lowercase types (string, integer, number, enum) - NO SQL DDL types.
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW wsp_text_rules AS
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   chunk_index,
# MAGIC   'text' AS extraction_source,
# MAGIC   ai_extract(
# MAGIC     chunk_text,
# MAGIC     '{
# MAGIC       "rules": {
# MAGIC         "type": "array",
# MAGIC         "description": "List of water access and extraction rules found in this text",
# MAGIC         "items": {
# MAGIC           "type": "object",
# MAGIC           "properties": {
# MAGIC             "rule_type": {
# MAGIC               "type": "enum",
# MAGIC               "labels": [
# MAGIC                 "extraction_limit",
# MAGIC                 "flow_class_threshold",
# MAGIC                 "access_condition",
# MAGIC                 "cease_to_pump",
# MAGIC                 "environmental_release",
# MAGIC                 "trading_rule",
# MAGIC                 "other"
# MAGIC               ],
# MAGIC               "description": "Type of water sharing rule"
# MAGIC             },
# MAGIC             "water_source": {
# MAGIC               "type": "string",
# MAGIC               "description": "The water source or management zone this rule applies to (e.g. Barwon River, Intersecting Streams)"
# MAGIC             },
# MAGIC             "condition": {
# MAGIC               "type": "string",
# MAGIC               "description": "The condition or trigger for this rule (e.g. when flow falls below X, during drought)"
# MAGIC             },
# MAGIC             "value": {
# MAGIC               "type": "number",
# MAGIC               "description": "Numeric value (flow rate, volume, or limit)"
# MAGIC             },
# MAGIC             "unit": {
# MAGIC               "type": "string",
# MAGIC               "description": "Unit of measurement (ML, GL, ML/day, ML/year, etc.)"
# MAGIC             },
# MAGIC             "applies_to": {
# MAGIC               "type": "string",
# MAGIC               "description": "Who or what this rule applies to (license category, all users, etc.)"
# MAGIC             },
# MAGIC             "section_reference": {
# MAGIC               "type": "string",
# MAGIC               "description": "Section or clause reference if mentioned in the text"
# MAGIC             }
# MAGIC           }
# MAGIC         }
# MAGIC       }
# MAGIC     }',
# MAGIC     MAP(
# MAGIC       'version', '2.1',
# MAGIC       'enableCitations', 'true',
# MAGIC       'enableConfidenceScores', 'true',
# MAGIC       'instructions', 'Extract all water access rules, extraction limits, flow thresholds, cease-to-pump conditions, and environmental requirements from NSW Water Sharing Plan text. Include numeric values and units where present.'
# MAGIC     )
# MAGIC   ) AS extracted
# MAGIC FROM workspace.default.wsp_chunks
# MAGIC WHERE
# MAGIC   -- Only process chunks likely to contain rules (filter by keywords to reduce cost)
# MAGIC   (
# MAGIC     lower(chunk_text) LIKE '%limit%'
# MAGIC     OR lower(chunk_text) LIKE '%flow%'
# MAGIC     OR lower(chunk_text) LIKE '%access%'
# MAGIC     OR lower(chunk_text) LIKE '%cease%'
# MAGIC     OR lower(chunk_text) LIKE '%threshold%'
# MAGIC     OR lower(chunk_text) LIKE '%extraction%'
# MAGIC     OR lower(chunk_text) LIKE '%environmental%'
# MAGIC     OR lower(chunk_text) LIKE '%compliance%'
# MAGIC   );
# MAGIC
# MAGIC -- Check extraction results: how many chunks had rules extracted, and were there errors?
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   count(*) AS chunks_processed,
# MAGIC   sum(CASE WHEN is_variant_null(extracted:error_message) THEN 1 ELSE 0 END) AS successful_extractions,
# MAGIC   sum(CASE WHEN NOT is_variant_null(extracted:error_message) THEN 1 ELSE 0 END) AS failed_extractions
# MAGIC FROM wsp_text_rules
# MAGIC GROUP BY plan_name
# MAGIC ORDER BY plan_name;
# MAGIC
# MAGIC -- Preview a few extractions
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   chunk_index,
# MAGIC   extracted:response:rules AS rules,
# MAGIC   size(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) AS rule_count
# MAGIC FROM wsp_text_rules
# MAGIC WHERE is_variant_null(extracted:error_message)
# MAGIC   AND size(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) > 0
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Step 14 — Union all rules into wsp_rules table
# MAGIC %sql
# MAGIC -- Union table-based rules and text-based rules into a single wsp_rules table
# MAGIC -- Explode the array of rules from ai_extract and flatten the v2.1 response structure
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.default.wsp_rules AS
# MAGIC -- Table rules (deterministic extraction)
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   extraction_source,
# MAGIC   rule_type,
# MAGIC   water_source,
# MAGIC   condition,
# MAGIC   value,
# MAGIC   unit,
# MAGIC   applies_to,
# MAGIC   section_reference,
# MAGIC   citation_ids,
# MAGIC   confidence_score
# MAGIC FROM wsp_table_rules
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC -- Text rules (ai_extract)
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   extraction_source,
# MAGIC   rule:rule_type:value::STRING AS rule_type,
# MAGIC   rule:water_source:value::STRING AS water_source,
# MAGIC   rule:condition:value::STRING AS condition,
# MAGIC   rule:value:value::DOUBLE AS value,
# MAGIC   rule:unit:value::STRING AS unit,
# MAGIC   rule:applies_to:value::STRING AS applies_to,
# MAGIC   rule:section_reference:value::STRING AS section_reference,
# MAGIC   try_cast(rule:rule_type:citation_ids AS ARRAY<INT>) AS citation_ids,
# MAGIC   rule:rule_type:confidence_score::DOUBLE AS confidence_score
# MAGIC FROM wsp_text_rules
# MAGIC LATERAL VIEW explode(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) AS rule
# MAGIC WHERE is_variant_null(extracted:error_message)
# MAGIC   AND rule IS NOT NULL;
# MAGIC
# MAGIC -- Basic data quality check
# MAGIC SELECT
# MAGIC   'Total rules' AS metric,
# MAGIC   count(*) AS count
# MAGIC FROM workspace.default.wsp_rules
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'Rules with confidence score' AS metric,
# MAGIC   count(*) AS count
# MAGIC FROM workspace.default.wsp_rules
# MAGIC WHERE confidence_score IS NOT NULL
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'Rules with numeric value' AS metric,
# MAGIC   count(*) AS count
# MAGIC FROM workspace.default.wsp_rules
# MAGIC WHERE value IS NOT NULL;

# COMMAND ----------

# DBTITLE 1,Step 15 — Preview: rule counts by plan, type, and source
# MAGIC %sql
# MAGIC -- Preview query: rule counts per plan, broken down by rule_type and extraction_source
# MAGIC
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   rule_type,
# MAGIC   extraction_source,
# MAGIC   count(*) AS rule_count,
# MAGIC   count(CASE WHEN value IS NOT NULL THEN 1 END) AS rules_with_numeric_value,
# MAGIC   avg(confidence_score) AS avg_confidence_score,
# MAGIC   min(confidence_score) AS min_confidence_score,
# MAGIC   max(confidence_score) AS max_confidence_score
# MAGIC FROM workspace.default.wsp_rules
# MAGIC GROUP BY plan_name, rule_type, extraction_source
# MAGIC ORDER BY plan_name, rule_count DESC;
# MAGIC
# MAGIC -- Sample rules from each category
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   rule_type,
# MAGIC   extraction_source,
# MAGIC   water_source,
# MAGIC   condition,
# MAGIC   value,
# MAGIC   unit,
# MAGIC   applies_to,
# MAGIC   round(confidence_score, 3) AS confidence
# MAGIC FROM workspace.default.wsp_rules
# MAGIC WHERE rule_type IS NOT NULL
# MAGIC ORDER BY plan_name, rule_type, confidence_score DESC NULLS LAST
# MAGIC LIMIT 50;

# COMMAND ----------

# DBTITLE 1,Step 13 — Extract rules from text chunks (ai_extract v2.1)
# MAGIC %sql
# MAGIC -- Extract structured rules from text chunks using ai_extract v2.1
# MAGIC -- with citations and confidence scores enabled.
# MAGIC -- Schema uses lowercase types (string, integer, number, enum) - NO SQL DDL types.
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW wsp_text_rules AS
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   chunk_index,
# MAGIC   'text' AS extraction_source,
# MAGIC   ai_extract(
# MAGIC     chunk_text,
# MAGIC     '{
# MAGIC       "rules": {
# MAGIC         "type": "array",
# MAGIC         "description": "List of water access and extraction rules found in this text",
# MAGIC         "items": {
# MAGIC           "type": "object",
# MAGIC           "properties": {
# MAGIC             "rule_type": {
# MAGIC               "type": "enum",
# MAGIC               "labels": [
# MAGIC                 "extraction_limit",
# MAGIC                 "flow_class_threshold",
# MAGIC                 "access_condition",
# MAGIC                 "cease_to_pump",
# MAGIC                 "environmental_release",
# MAGIC                 "trading_rule",
# MAGIC                 "other"
# MAGIC               ],
# MAGIC               "description": "Type of water sharing rule"
# MAGIC             },
# MAGIC             "water_source": {
# MAGIC               "type": "string",
# MAGIC               "description": "The water source or management zone this rule applies to (e.g. Barwon River, Intersecting Streams)"
# MAGIC             },
# MAGIC             "condition": {
# MAGIC               "type": "string",
# MAGIC               "description": "The condition or trigger for this rule (e.g. when flow falls below X, during drought)"
# MAGIC             },
# MAGIC             "value": {
# MAGIC               "type": "number",
# MAGIC               "description": "Numeric value (flow rate, volume, or limit)"
# MAGIC             },
# MAGIC             "unit": {
# MAGIC               "type": "string",
# MAGIC               "description": "Unit of measurement (ML, GL, ML/day, ML/year, etc.)"
# MAGIC             },
# MAGIC             "applies_to": {
# MAGIC               "type": "string",
# MAGIC               "description": "Who or what this rule applies to (license category, all users, etc.)"
# MAGIC             },
# MAGIC             "section_reference": {
# MAGIC               "type": "string",
# MAGIC               "description": "Section or clause reference if mentioned in the text"
# MAGIC             }
# MAGIC           }
# MAGIC         }
# MAGIC       }
# MAGIC     }',
# MAGIC     MAP(
# MAGIC       'version', '2.1',
# MAGIC       'enableCitations', 'true',
# MAGIC       'enableConfidenceScores', 'true',
# MAGIC       'instructions', 'Extract all water access rules, extraction limits, flow thresholds, cease-to-pump conditions, and environmental requirements from NSW Water Sharing Plan text. Include numeric values and units where present.'
# MAGIC     )
# MAGIC   ) AS extracted
# MAGIC FROM workspace.default.wsp_chunks
# MAGIC WHERE
# MAGIC   -- Only process chunks likely to contain rules (filter by keywords to reduce cost)
# MAGIC   (
# MAGIC     lower(chunk_text) LIKE '%limit%'
# MAGIC     OR lower(chunk_text) LIKE '%flow%'
# MAGIC     OR lower(chunk_text) LIKE '%access%'
# MAGIC     OR lower(chunk_text) LIKE '%cease%'
# MAGIC     OR lower(chunk_text) LIKE '%threshold%'
# MAGIC     OR lower(chunk_text) LIKE '%extraction%'
# MAGIC     OR lower(chunk_text) LIKE '%environmental%'
# MAGIC     OR lower(chunk_text) LIKE '%compliance%'
# MAGIC   );
# MAGIC
# MAGIC -- Check extraction results: how many chunks had rules extracted, and were there errors?
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   count(*) AS chunks_processed,
# MAGIC   sum(CASE WHEN is_variant_null(extracted:error_message) THEN 1 ELSE 0 END) AS successful_extractions,
# MAGIC   sum(CASE WHEN NOT is_variant_null(extracted:error_message) THEN 1 ELSE 0 END) AS failed_extractions
# MAGIC FROM wsp_text_rules
# MAGIC GROUP BY plan_name
# MAGIC ORDER BY plan_name;
# MAGIC
# MAGIC -- Preview a few extractions
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   chunk_index,
# MAGIC   extracted:response:rules AS rules,
# MAGIC   size(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) AS rule_count
# MAGIC FROM wsp_text_rules
# MAGIC WHERE is_variant_null(extracted:error_message)
# MAGIC   AND size(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) > 0
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Step 14 — Union all rules into wsp_rules table
# MAGIC %sql
# MAGIC -- Union table-based rules and text-based rules into a single wsp_rules table
# MAGIC -- Explode the array of rules from ai_extract and flatten the v2.1 response structure
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.default.wsp_rules AS
# MAGIC -- Table rules (deterministic extraction)
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   extraction_source,
# MAGIC   rule_type,
# MAGIC   water_source,
# MAGIC   condition,
# MAGIC   value,
# MAGIC   unit,
# MAGIC   applies_to,
# MAGIC   section_reference,
# MAGIC   citation_ids,
# MAGIC   confidence_score
# MAGIC FROM wsp_table_rules
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC -- Text rules (ai_extract)
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   extraction_source,
# MAGIC   rule:rule_type:value::STRING AS rule_type,
# MAGIC   rule:water_source:value::STRING AS water_source,
# MAGIC   rule:condition:value::STRING AS condition,
# MAGIC   rule:value:value::DOUBLE AS value,
# MAGIC   rule:unit:value::STRING AS unit,
# MAGIC   rule:applies_to:value::STRING AS applies_to,
# MAGIC   rule:section_reference:value::STRING AS section_reference,
# MAGIC   try_cast(rule:rule_type:citation_ids AS ARRAY<INT>) AS citation_ids,
# MAGIC   rule:rule_type:confidence_score::DOUBLE AS confidence_score
# MAGIC FROM wsp_text_rules
# MAGIC LATERAL VIEW explode(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) AS rule
# MAGIC WHERE is_variant_null(extracted:error_message)
# MAGIC   AND rule IS NOT NULL;
# MAGIC
# MAGIC -- Basic data quality check
# MAGIC SELECT
# MAGIC   'Total rules' AS metric,
# MAGIC   count(*) AS count
# MAGIC FROM workspace.default.wsp_rules
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'Rules with confidence score' AS metric,
# MAGIC   count(*) AS count
# MAGIC FROM workspace.default.wsp_rules
# MAGIC WHERE confidence_score IS NOT NULL
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'Rules with numeric value' AS metric,
# MAGIC   count(*) AS count
# MAGIC FROM workspace.default.wsp_rules
# MAGIC WHERE value IS NOT NULL;

# COMMAND ----------

# DBTITLE 1,Step 15 — Preview: rule counts by plan, type, and source
# MAGIC %sql
# MAGIC -- Preview query: rule counts per plan, broken down by rule_type and extraction_source
# MAGIC
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   rule_type,
# MAGIC   extraction_source,
# MAGIC   count(*) AS rule_count,
# MAGIC   count(CASE WHEN value IS NOT NULL THEN 1 END) AS rules_with_numeric_value,
# MAGIC   avg(confidence_score) AS avg_confidence_score,
# MAGIC   min(confidence_score) AS min_confidence_score,
# MAGIC   max(confidence_score) AS max_confidence_score
# MAGIC FROM workspace.default.wsp_rules
# MAGIC GROUP BY plan_name, rule_type, extraction_source
# MAGIC ORDER BY plan_name, rule_count DESC;
# MAGIC
# MAGIC -- Sample rules from each category
# MAGIC SELECT
# MAGIC   plan_name,
# MAGIC   rule_type,
# MAGIC   extraction_source,
# MAGIC   water_source,
# MAGIC   condition,
# MAGIC   value,
# MAGIC   unit,
# MAGIC   applies_to,
# MAGIC   round(confidence_score, 3) AS confidence
# MAGIC FROM workspace.default.wsp_rules
# MAGIC WHERE rule_type IS NOT NULL
# MAGIC ORDER BY plan_name, rule_type, confidence_score DESC NULLS LAST
# MAGIC --LIMIT 50;

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE system.ai_gateway.usage
# MAGIC