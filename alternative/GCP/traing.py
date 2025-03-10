from google.cloud import aiplatform

# Initialize Vertex AI with your project and region.
aiplatform.init(project="YOUR_PROJECT_ID", location="YOUR_REGION")

job = aiplatform.PythonPackageTrainingJob(
    display_name="finetune-llama2-lora",
    python_package_gcs_uri="gs://YOUR_BUCKET/path/to/training_package/my_training_package-0.1.tar.gz",
    python_module_name="trainer.run_qa_lora",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch:2.1.0",  # Using a PyTorch container
    requirements=[
        "transformers==4.37.0",
        "datasets",
        "peft",
        "torch==2.1.0"
    ],
)

job.run(
    args=[
        "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
        "--train_file", "gs://YOUR_BUCKET/path/to/synthetic_qa.jsonl",
        "--output_dir", "gs://YOUR_BUCKET/path/to/output"
    ],
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=1,
    base_output_dir="gs://YOUR_BUCKET/path/to/output",
    sync=True
)
