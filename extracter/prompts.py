FIGURE_PROMPT_TEMPLATE = """
You are analyzing a figure extracted from an academic PDF for a multimodal RAG system.
Return ONLY valid JSON. Do not use markdown or code fences.
Classify the image based on full visual structure, not only text blocks.
If uncertain, classify it as "figure".

Return exactly this JSON:
{{
  "image_type": "pipeline_diagram|flowchart|architecture_diagram|chart|table_screenshot|photo|equation|figure|other",
  "classification": "short specific category",
  "short_description": "one sentence",
  "detailed_description": "2 to 4 sentences describing the image accurately",
  "visible_text_summary": "summary of important readable text, or not_readable",
  "caption_summary": "summary of the PDF caption, or empty string",
  "rag_search_text": "one useful paragraph for semantic search",
  "rag_keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
  "entities": ["entity1", "entity2"],
  "should_index_for_rag": true
}}

PDF caption:
{caption}
""".strip()

TABLE_PROMPT_TEMPLATE = """
You are analyzing a table image extracted from a PDF for a RAG system.

Return ONLY valid JSON.
Do not use markdown outside JSON.
Do not use code fences.
Do not use newline characters inside JSON string values.
Do not extract every table cell.
Summarize the table instead.

Return exactly this JSON shape:
{{
  "table_type": "comparison_table|results_table|ablation_table|dataset_table|metrics_table|other",
  "short_description": "one sentence",
  "columns_summary": "short summary of the table columns",
  "key_findings": ["finding1", "finding2", "finding3"],
  "visible_text_long_summary": "long summary of readable table text",
  "visible_text_summary": "short summary of readable table text",
  "caption_summary": "summary of the PDF caption, or empty string",
  "rag_search_text": "one clean paragraph useful for semantic search",
  "rag_keywords": ["keyword1", "keyword2", "keyword3"],
  "should_index_for_rag": true
}}

PDF caption:
{caption}
""".strip()
