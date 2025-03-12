# run_pipeline.py
import subprocess

subprocess.run(["python3", "1_read_docs_mindsDB.py"])
subprocess.run(["python3", "2_clean_the_docs.py"])
subprocess.run(["python3", "3_chunk_the_docs.py"])
subprocess.run(["python3", "4_convert_docs_2_QA_parallel.py"])
subprocess.run(["python3", "5_train_on_colab.py"])
