# Unique task identifier
TASK_ID="6be4e624-ace3-421c-9719-c335f9d14843"

# Base model to fine-tune (from HuggingFace)
MODEL="GraydientPlatformAPI/albedobase2-xl"

# Dataset ZIP file location (must be a ZIP file with images)
DATASET_ZIP="https://gradients.s3.eu-north-1.amazonaws.com/3eefc53d41e219b4_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20251110%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251110T102438Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=180f3ae736cf78b23a7d12a4f57b14c7c2f522256379068eeac7c376749d3c52"

# Model type: "sdxl" or "flux"
MODEL_TYPE="sdxl"

# Optional: Repository name for the trained model
EXPECTED_REPO_NAME="6be4e624-ace3-421c-9719-c335f9d14843"

# For uploading the outputs
HUGGINGFACE_TOKEN="Your Huggingface Token"
HUGGINGFACE_USERNAME="Your Huggingface Username"
EXPECTED_REPO_NAME="fd97f02f-82c5-40dd-bee7-73ff18fcf39c"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

# Build the downloader image
# docker build --no-cache -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
# docker build --no-cache -t standalone-image-trainer -f dockerfiles/standalone-image-trainer.dockerfile .

# Build the hf uploader image
# docker build --no-cache -t hf-uploader -f dockerfiles/hf-uploader.dockerfile .

# Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET_ZIP" \
  --task-type "ImageTask"

# Run the training
echo "Starting image training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=32g \
  --cpus=8 \
  --network none \
  --env TRANSFORMERS_CACHE=/cache/hf_cache \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name image-trainer-example \
  standalone-image-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset-zip "$DATASET_ZIP" \
  --model-type "$MODEL_TYPE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --hours-to-complete 1 \
  --reg-ratio 0.98

echo "Uploading model to HuggingFace..."
docker run --rm --gpus all \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  --env TASK_ID="$TASK_ID" \
  --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  --env LOCAL_FOLDER="$LOCAL_FOLDER" \
  --env HF_REPO_SUBFOLDER="checkpoints" \
  --name hf-uploader \
  hf-uploader
