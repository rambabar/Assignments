import requests
import base64

# Encode image data to Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Base64 encoded images
image1_base64 = image_to_base64('image4.jpg')
image2_base64 = image_to_base64('image4.jpg')

# API endpoint URL
url = 'http://localhost:8000/process_images'
# url = 'https://fde3-34-86-24-224.ngrok.io/process_images'
# url = 'https://image-matching-mmu6qoex5q-uc.a.run.app/process_images'

# Request payload
payload = {
    'image1': image1_base64,
    'image2': image2_base64
}

# Send POST request
response = requests.post(url, json=payload)

# Check response status code
if response.status_code == 200:
    # Get the output image in Base64 format
    output_image_base64 = response.json()['output_image']
    
    # Decode the output image from Base64
    output_image_data = base64.b64decode(output_image_base64)

    # Save the output image to a file
    with open('output_image.jpg', 'wb') as file:
        file.write(output_image_data)

    print("Output image saved successfully.")
else:
    print("Error:", response.status_code, response.text)
