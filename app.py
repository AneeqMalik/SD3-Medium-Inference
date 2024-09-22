import os
from flask import Flask, request, jsonify
from io import BytesIO

from sd3.sd3_t4_pipeline import load_sd3_t4_pipeline, generate_sd3_t4_image
from huggingface_hub import login
import boto3
from dotenv import load_dotenv
import json


load_dotenv()

# Use environment variable for Hugging Face token
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables")
login(token=hf_token)

app = Flask(__name__)

# Load the pipeline once the application starts.
model = load_sd3_t4_pipeline()

# Setup boto3 session
aws_access_key_id = os.environ.get('aws_access_key_id')
aws_secret_access_key = os.environ.get('aws_secret_access_key')
region_name = os.environ.get('region_name')
if not aws_access_key_id or not aws_secret_access_key or not region_name:
    raise ValueError("AWS credentials or region not found in environment variables")

boto_session = boto3.setup_default_session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)
s3_client = boto3.client('s3')

# Define an endpoint for health check
@app.route('/ping', methods=['GET'])
def ping():
  return '', 200

# Define an endpoint for making predictions
@app.route('/invocations', methods=['POST'])
def predict():
    """
    Function which responds to the invocations requests.
    """
   # Get data from the POST request
    data = request.get_data().decode('utf-8')
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    negative_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 28)
    guidance_scale = data.get("guidance_scale", 7)
    num_images_per_prompt = data.get("num_images_per_prompt", 1)

    # Generate images
    generated_images = generate_sd3_t4_image(
        prompt,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt,
    )

    # Upload images to S3 and collect URLs
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        return jsonify({"error": "Bucket name not found in environment variables"}), 500

    image_urls = []
    for idx, image in enumerate(generated_images):
        try:
            image_buffer = BytesIO()
            image.save(image_buffer, format='PNG')
            image_buffer.seek(0)
            image_key = f"generated_images/{prompt.replace(' ', '_')}_{idx}.png"
            s3_client.upload_fileobj(image_buffer, bucket_name, image_key)
            image_url = f"https://{bucket_name}.s3.amazonaws.com/{image_key}"
            image_urls.append(image_url)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return json.dumps({"image_urls": image_urls})