# Image Matching API using Kornia Library

This folder contains the API code for image matching using the Kornia library. The API allows you to perform image matching and extract matching keypoints between two input images.

## Installation

To run the API, make sure you have the following dependencies installed:

- Python 3.9 or higher
- Kornia library
- Additional dependencies specified in the `requirements.txt` file

You can install the dependencies using the following command:


`
pip install -r requirements.txt
`

## Usage

1. Place the input images in the desired location or directory.
2. Start the API by running the following command:

`
python main.py
`

3. The API will start running and provide a RESTful interface to interact with.
4. Use your preferred HTTP client to send a POST request to the API endpoint with the input images' paths or URLs.
5. The API will process the images, perform image matching using the Kornia library, and return the output image in base64 format.

## API Endpoints

The API provides the following endpoint:

### POST /image/match

- Description: Perform image matching between two input images.
- Request Body:

`
{
"image1": "path_to_image1",
"image2": "path_to_image2"
}
`

- Response Body:
`
{
"output_image": "base64_encoded_output_image"
}
`

- Example:

`
curl -X POST -H "Content-Type: application/json" -d '{
"image1": "path_to_image1",
"image2": "path_to_image2"
}' http://localhost:8000/process_images
`

To run the Docker image for the Kornia Image Matching application, you need to follow these steps:
1. Pull the Docker image from Docker Hub by running the following command:
`
docker pull rambabar108/kornia_image_matching:v1.0
`

2. Run the Docker container using the pulled image by executing the following command:
`
docker run -p 8000:8000 rambabar108/kornia_image_matching:v1.0
`

This command starts the Docker container and maps port 8000 on your host machine to port 8000 inside the container. The application running inside the container will be accessible through localhost:8000 in your web browser.
Ensure that Docker is installed and running on your machine before executing these commands.
