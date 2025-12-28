# Enriched PyPI Dataset

Each JSON file corresponds to one PyPI package.

clean_readme.py cleans readme's.
chunk_readme.py chunks readme's.
add_tags.py adds meta tags. 

The resulting dataset is given in the enriched folder. 
It contains ~370 libraries with at least 500,000 monthly downloads 
and a minimal readme length of 200 based on appearance of 
DS_CLASSIFIER_KEYWORDS in select_packages.py

Structure: 
- name: package name
- summary: PyPI summary
- language: always "python"
- tags: heuristic semantic tags
- chunks: cleaned semantic text chunks

Chunks from the same file always belong to the same library
and should be aggregated at the library level when needed.

A chunk contains at least 200 words, and at most 500 words with 
one exception. If the total readme has less than 800 words, the 
readme is not chunked. 

Tags are derived from two complementary sources: PyPI metadata 
and the cleaned README text. First, PyPI classifiers are mapped 
to tags. Second, the README text is scanned for domain-specific 
keywords, and a tag is assigned when multiple related terms appear. 
The final tag set is the union of both sources and is stored 
alongside the library’s chunks.

Possible tags are one or more of {web, data, ml, math, visualization, cli, ui, dev}.
