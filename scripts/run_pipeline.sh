#!/bin/bash

# Clone the MindsDB documentation repository
git clone https://github.com/mindsdb/mindsdb-docs.git

# Navigate to the scripts directory
cd mindsdb-docs/scripts

# run_pipeline.sh
#!/bin/bash

python3 1_read_docs_mindsDB.py
python3 2_clean_the_docs.py
python3 3_chunk_the_docs.py
python3 4_convert_docs_2_QA_parallel.py
python3 5_train_on_colab.py

