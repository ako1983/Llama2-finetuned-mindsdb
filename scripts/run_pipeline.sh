#!/bin/bash

# Clone the MindsDB documentation repository
git clone https://github.com/mindsdb/mindsdb-docs.git

# Navigate to the scripts directory
cd mindsdb-docs/scripts

# Execute the scripts in sequence
./read_docs.sh
# Add other script executions as needed
./train_on_colab.sh
