import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import uuid
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load the YOLOv8 model
model = YOLO('deepfashion2_yolov8s-seg.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/results/<filename>')
def result_file(filename):
    response = send_from_directory(app.config['RESULTS_FOLDER'], filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/segment', methods=['POST'])
def segment_clothing():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run YOLOv8 segmentation
        results = model(filepath, conf=0.25)
        
        # Process results
        result_images = []
        
        for i, result in enumerate(results):
            if len(result.masks) == 0:
                continue
                
            original_image = Image.open(filepath)
            original_image_np = np.array(original_image)
            
            for j, mask in enumerate(result.masks):
                # Convert mask to binary format suitable for masking
                mask_np = mask.data.cpu().numpy()[0]
                mask_np = cv2.resize(mask_np, (original_image_np.shape[1], original_image_np.shape[0]))
                binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                
                # Apply mask to original image
                masked_image_np = original_image_np.copy()
                for c in range(3):  # Apply mask to each channel
                    masked_image_np[:, :, c] = masked_image_np[:, :, c] * (binary_mask / 255)
                
                # Find bounding box coordinates of mask
                y_indices, x_indices = np.where(binary_mask > 0)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                
                # Crop the image to the mask's bounding box
                cropped_mask = masked_image_np[y_min:y_max, x_min:x_max]
                
                # Save the result
                result_filename = f"result_{uuid.uuid4()}.png"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_path, cv2.cvtColor(cropped_mask, cv2.COLOR_RGB2BGR))
                
                # Add to results
                result_images.append({
                    'filename': result_filename,
                    'class': result.names[int(result.boxes.cls[j])],
                    'confidence': float(result.boxes.conf[j])
                })
        
        return jsonify({
            'original': unique_filename,
            'results': result_images
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/clean-segmentation', methods=['POST'])
def clean_segmentation():
    # Get image path from request
    data = request.json
    if not data or 'image_filename' not in data:
        return jsonify({'success': False, 'error': 'No image specified'})
    
    image_filename = data['image_filename']
    cleanup_method = data.get('cleanup_method', 'remove_background')
    category = data.get('category', '')
    
    # Validate the image exists
    image_path = os.path.join(app.config['RESULTS_FOLDER'], image_filename)
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'Image not found'})
    
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to load image'})
        
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply the requested cleanup method
        if cleanup_method == 'remove_background':
            # Create a transparent background version
            # For this demo, we'll simulate this by creating a mask from non-black pixels
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Create a 4-channel RGBA image
            height, width = image_rgb.shape[:2]
            image_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            image_rgba[:, :, :3] = image_rgb
            image_rgba[:, :, 3] = mask
            
            # Save as PNG to preserve transparency
            cleaned_filename = f"cleaned_{uuid.uuid4()}.png"
            cleaned_path = os.path.join(app.config['RESULTS_FOLDER'], cleaned_filename)
            cv2.imwrite(cleaned_path, cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
        
        elif cleanup_method == 'enhance_contrast':
            # Enhance contrast and brightness
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Save the enhanced image
            cleaned_filename = f"cleaned_{uuid.uuid4()}.png"
            cleaned_path = os.path.join(app.config['RESULTS_FOLDER'], cleaned_filename)
            cv2.imwrite(cleaned_path, cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
        
        elif cleanup_method == 'smooth_edges':
            # Apply morphological operations to smooth edges
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to smooth the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply the smoothed mask to the original image
            smoothed_image = image_rgb.copy()
            for c in range(3):
                smoothed_image[:, :, c] = smoothed_image[:, :, c] * (smoothed_mask / 255)
            
            # Save the smoothed image
            cleaned_filename = f"cleaned_{uuid.uuid4()}.png"
            cleaned_path = os.path.join(app.config['RESULTS_FOLDER'], cleaned_filename)
            cv2.imwrite(cleaned_path, cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR))
        
        else:
            return jsonify({'success': False, 'error': 'Invalid cleanup method'})
        
        # If a category is provided, also save to the database folder
        if category in ['UpperClothes', 'Bottoms']:
            os.makedirs(f'database/{category}', exist_ok=True)
            db_filename = f"cleaned_{uuid.uuid4()}.png"
            db_path = os.path.join(f'database/{category}', db_filename)
            shutil.copy(cleaned_path, db_path)
            
            return jsonify({
                'success': True, 
                'cleaned_filename': cleaned_filename,
                'db_filename': db_filename,
                'category': category
            })
        
        return jsonify({'success': True, 'cleaned_filename': cleaned_filename})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/erase-image-part', methods=['POST'])
def erase_image_part():
    data = request.json
    if not data or 'image_filename' not in data or 'erase_data' not in data:
        return jsonify({'success': False, 'error': 'Missing required data'})
    
    image_filename = data['image_filename']
    erase_data = data['erase_data']
    category = data.get('category', '')  # Optional category for saving to wardrobe
    
    # Validate the image exists
    image_path = os.path.join(app.config['RESULTS_FOLDER'], image_filename)
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'Image not found'})
    
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to load image'})
        
        # Check if image has alpha channel, if not add one
        if image.shape[2] == 3:
            # Create alpha channel (fully opaque)
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            image = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], alpha))
        
        # Create a mask for all erasure points
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply erasures
        for point in erase_data:
            x, y, radius = point['x'], point['y'], point['radius']
            # Convert coordinates from percentage to pixels
            x_px = int(x * image.shape[1])
            y_px = int(y * image.shape[0])
            radius_px = int(radius * max(image.shape[0], image.shape[1]))
            
            # Add each eraser circle to the mask
            cv2.circle(mask, (x_px, y_px), radius_px, 255, -1)
        
        # Apply the mask to the alpha channel with anti-aliasing
        mask_float = mask.astype(np.float32) / 255.0
        # Blur the mask slightly for smoother edges
        mask_float = cv2.GaussianBlur(mask_float, (5, 5), 0)
        
        # Where mask is 1, set alpha to 0 (transparent)
        image[:, :, 3] = image[:, :, 3] * (1.0 - mask_float)
        image[:, :, 3] = image[:, :, 3].astype(np.uint8)
        
        # Save the erased image to results folder
        erased_filename = f"erased_{uuid.uuid4()}.png"
        erased_path = os.path.join(app.config['RESULTS_FOLDER'], erased_filename)
        cv2.imwrite(erased_path, image)
        
        # If a category is provided, also save to the database folder
        db_filename = None
        if category in ['UpperClothes', 'Bottoms']:
            os.makedirs(f'database/{category}', exist_ok=True)
            db_filename = f"erased_{uuid.uuid4()}.png"
            db_path = os.path.join(f'database/{category}', db_filename)
            cv2.imwrite(db_path, image)
            
            return jsonify({
                'success': True, 
                'erased_filename': erased_filename,
                'db_filename': db_filename,
                'category': category
            })
        
        return jsonify({'success': True, 'erased_filename': erased_filename})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/outfit-pairing')
def outfit_pairing():
    # Get all upper clothes
    upper_clothes = []
    for filename in os.listdir('database/UpperClothes'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.avif')):
            upper_clothes.append({
                'filename': filename,
                'path': f'database/UpperClothes/{filename}'
            })
    
    # Get all bottoms
    bottoms = []
    for filename in os.listdir('database/Bottoms'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.avif')):
            bottoms.append({
                'filename': filename,
                'path': f'database/Bottoms/{filename}'
            })
    
    return render_template('outfit_pairing.html', upper_clothes=upper_clothes, bottoms=bottoms)

@app.route('/database/<folder>/<filename>')
def database_file(folder, filename):
    return send_from_directory(f'database/{folder}', filename)

@app.route('/add_to_wardrobe', methods=['POST'])
def add_to_wardrobe():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    category = request.form.get('category')
    enhance_option = request.form.get('enhance_option', 'original')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if not category:
        return jsonify({'success': False, 'error': 'No category selected'})
    
    if category not in ['UpperClothes', 'Bottoms']:
        return jsonify({'success': False, 'error': 'Invalid category'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create database directory if it doesn't exist
        os.makedirs(f'database/{category}', exist_ok=True)
        
        # Save the file to the appropriate folder
        filepath = os.path.join(f'database/{category}', filename)
        
        # In a real app, we would apply different AI enhancements based on the enhance_option
        # For this demo, we'll just save the original file
        
        # If enhance_option is not 'original', we would call an AI API here
        # For example with the 'enhance' option:
        # 1. Remove background
        # 2. Enhance image quality
        # 3. Normalize sizing
        
        # For 'similar' option:
        # 1. Use image generation API to create a similar item
        # 2. Save the generated image instead
        
        # For 'ideal' option:
        # 1. Clean up image
        # 2. Create perfect representation of the item
        
        # For now, we'll just save the original file
        file.save(filepath)
        
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/delete-wardrobe-item', methods=['POST'])
def delete_wardrobe_item():
    data = request.json
    
    if not data or 'filename' not in data or 'category' not in data:
        return jsonify({'success': False, 'error': 'Missing required data'})
    
    filename = data['filename']
    category = data['category']
    
    # Validate the category
    if category not in ['UpperClothes', 'Bottoms']:
        return jsonify({'success': False, 'error': 'Invalid category'})
    
    # Validate the filename against directory traversal attacks
    if '..' in filename or '/' in filename:
        return jsonify({'success': False, 'error': 'Invalid filename'})
    
    # Check if the file exists
    filepath = os.path.join(f'database/{category}', filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        # Delete the file
        os.remove(filepath)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate-outfit-comparison', methods=['POST'])
def generate_outfit_comparison():
    data = request.json
    
    if not data or 'outfits' not in data or len(data['outfits']) != 2:
        return jsonify({'success': False, 'error': 'Need exactly two outfits to compare'})
    
    try:
        outfits = data['outfits']
        
        # Create a canvas for the comparison image
        # Size for a typical social media image (1200 x 630)
        canvas_width = 1200
        canvas_height = 630
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background
        
        # Add a nice background color gradient
        for y in range(canvas_height):
            color_factor = y / canvas_height
            bg_color = [
                255 - int(color_factor * 10),  # Red
                255 - int(color_factor * 20),  # Green
                255 - int(color_factor * 30)   # Blue
            ]
            canvas[y, :] = bg_color
        
        # Add "VS" text in the middle instead of a line
        font = cv2.FONT_HERSHEY_SIMPLEX
        vs_text = "VS"
        
        # Get text size for centering
        text_size = cv2.getTextSize(vs_text, font, 2.5, 5)[0]
        text_x = (canvas_width - text_size[0]) // 2
        text_y = canvas_height // 2 + text_size[1] // 2
        
        # Draw VS text with a subtle shadow effect
        # Shadow
        cv2.putText(canvas, vs_text, (text_x + 3, text_y + 3), font, 2.5, (100, 100, 100), 5, cv2.LINE_AA)
        # Main text
        cv2.putText(canvas, vs_text, (text_x, text_y), font, 2.5, (231, 84, 128), 5, cv2.LINE_AA)
        
        # Process each outfit
        for i, outfit in enumerate(outfits):
            # Load upper and bottom images
            upper_path = os.path.join('database/UpperClothes', outfit['upper_filename'])
            bottom_path = os.path.join('database/Bottoms', outfit['bottom_filename'])
            
            if os.path.exists(upper_path) and os.path.exists(bottom_path):
                upper_img = cv2.imread(upper_path, cv2.IMREAD_UNCHANGED)
                bottom_img = cv2.imread(bottom_path, cv2.IMREAD_UNCHANGED)
                
                # Process images if they have transparency (4 channels)
                if upper_img.shape[2] == 4:
                    # Convert BGRA to BGR with white background
                    alpha = upper_img[:, :, 3] / 255.0
                    upper_img_rgb = upper_img[:, :, :3]
                    white_bg = np.ones_like(upper_img_rgb) * 255
                    upper_img = (alpha[:, :, np.newaxis] * upper_img_rgb + 
                                (1 - alpha[:, :, np.newaxis]) * white_bg).astype(np.uint8)
                
                if bottom_img.shape[2] == 4:
                    # Convert BGRA to BGR with white background
                    alpha = bottom_img[:, :, 3] / 255.0
                    bottom_img_rgb = bottom_img[:, :, :3]
                    white_bg = np.ones_like(bottom_img_rgb) * 255
                    bottom_img = (alpha[:, :, np.newaxis] * bottom_img_rgb + 
                                 (1 - alpha[:, :, np.newaxis]) * white_bg).astype(np.uint8)
                
                # Resize images to fit the canvas
                max_item_width = (canvas_width // 2) - 60  # Half width minus margins
                
                # Resize upper image
                upper_height, upper_width = upper_img.shape[:2]
                upper_scale = min(max_item_width / upper_width, 200 / upper_height)
                upper_resized = cv2.resize(upper_img, (int(upper_width * upper_scale), int(upper_height * upper_scale)))
                
                # Resize bottom image
                bottom_height, bottom_width = bottom_img.shape[:2]
                bottom_scale = min(max_item_width / bottom_width, 280 / bottom_height)
                bottom_resized = cv2.resize(bottom_img, (int(bottom_width * bottom_scale), int(bottom_height * bottom_scale)))
                
                # Calculate positions for left or right side
                if i == 0:  # Left side
                    x_offset = (canvas_width // 4) - (upper_resized.shape[1] // 2)
                else:  # Right side
                    x_offset = (canvas_width // 4 * 3) - (upper_resized.shape[1] // 2)
                
                # Place upper item
                y_upper = 100
                upper_roi = canvas[y_upper:y_upper + upper_resized.shape[0], x_offset:x_offset + upper_resized.shape[1]]
                if upper_roi.shape[:2] == upper_resized.shape[:2]:
                    canvas[y_upper:y_upper + upper_resized.shape[0], x_offset:x_offset + upper_resized.shape[1]] = upper_resized
                
                # Place bottom item
                y_bottom = y_upper + upper_resized.shape[0] + 20
                bottom_roi = canvas[y_bottom:y_bottom + bottom_resized.shape[0], x_offset:x_offset + bottom_resized.shape[1]]
                if bottom_roi.shape[:2] == bottom_resized.shape[:2]:
                    canvas[y_bottom:y_bottom + bottom_resized.shape[0], x_offset:x_offset + bottom_resized.shape[1]] = bottom_resized
                
                # Add outfit labels
                if 'style' in outfit and outfit['style']:
                    label_y = y_bottom + bottom_resized.shape[0] + 30
                    cv2.putText(canvas, outfit['style'].capitalize(), 
                                (x_offset, label_y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                
                if 'event' in outfit and outfit['event']:
                    label_y = y_bottom + bottom_resized.shape[0] + 60
                    event_text = outfit['event'].replace('-', ' ').capitalize()
                    cv2.putText(canvas, event_text, 
                                (x_offset, label_y), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Add rating stars if available
                if 'rating' in outfit:
                    stars_y = y_bottom + bottom_resized.shape[0] + 90
                    for j in range(5):
                        star_color = (255, 215, 0) if j < outfit['rating'] else (200, 200, 200)  # Gold or gray
                        cv2.putText(canvas, "★", (x_offset + j*25, stars_y), font, 1, star_color, 2, cv2.LINE_AA)
        
        # Add a subtle watermark
        watermark = "Created with AI Stylist"
        cv2.putText(canvas, watermark, (canvas_width - 250, canvas_height - 20), 
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Save the comparison image
        comparison_filename = f"comparison_{uuid.uuid4()}.jpg"
        comparison_path = os.path.join(app.config['RESULTS_FOLDER'], comparison_filename)
        cv2.imwrite(comparison_path, canvas)
        
        return jsonify({
            'success': True,
            'comparison_filename': comparison_filename,
            'url': f'/results/{comparison_filename}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 