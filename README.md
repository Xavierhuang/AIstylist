# Clothing Segmentation Web App

A simple Flask web application that allows users to upload images of clothing items. The app uses YOLOv8 with the DeepFashion2 model to segment and crop the clothing items from the images.

## Features

- Upload images of clothing items
- Automatic segmentation using YOLOv8 with DeepFashion2 model
- Crop and extract each clothing item detected
- Display the segmented results with class prediction and confidence score

## Requirements

- Python 3.8+
- Flask
- Ultralytics YOLOv8
- OpenCV
- Pillow
- NumPy

## Setup

1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Make sure you have the DeepFashion2 YOLOv8 model file (`deepfashion2_yolov8s-seg.pt`) in the root directory.

## Running the Application

1. Start the Flask development server:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Use the interface to upload an image and view the segmentation results.

## Folder Structure

- `app.py`: Main Flask application
- `templates/`: Contains HTML templates
- `uploads/`: Stores uploaded images
- `results/`: Stores segmentation results
- `deepfashion2_yolov8s-seg.pt`: YOLOv8 model file for clothing segmentation

## Note

This application is for demonstration purposes. The segmentation results depend on the quality of the input images and the performance of the YOLOv8 model. 