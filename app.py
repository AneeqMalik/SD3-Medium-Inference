import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

app = FastAPI()

# Load the pipeline once the application starts.
load_sd3_t4_pipeline()

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
@app.get('/ping')
async def ping():
    return 'The server is running OK!'

# Define an endpoint for making predictions
@app.post('/invocations')
async def predict(request: Request):
    """
    Function which responds to the invocations requests.
    """
    # Get data from the POST request
    data = await request.json()
    prompt = data.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    negative_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 28)
    guidance_scale = data.get("guidance_scale", 7)

    # Generate images
    generated_images = generate_sd3_t4_image(
        prompt,
        negative_prompt,
        num_inference_steps,
        guidance_scale,
    )

    # Upload images to S3 and collect URLs
    bucket_name = os.environ.get('bucket_name')
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Bucket name not found in environment variables")

    image_urls = []
    for idx, image in enumerate(generated_images):
        try:
            image_buffer = BytesIO()
            image.save(image_buffer, format='PNG')
            image_buffer.seek(0)
            image_key = f"generated_images/{prompt.replace(' ', '_')}_{idx}.png"
            s3_client.upload_fileobj(image_buffer, bucket_name, image_key)
            
            # Generate a pre-signed URL for the uploaded image
            image_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': image_key},
                ExpiresIn=3600  # URL expiration time in seconds
            )
            image_urls.append(image_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"image_urls": image_urls})