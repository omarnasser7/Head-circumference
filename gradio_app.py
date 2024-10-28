import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model

# Enhanced CSS with background and additional styling
custom_css = """
body {
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) fixed !important;
}

.container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 2rem !important;
}

.gr-interface {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.gr-button {
    background: linear-gradient(135deg, #1976D2 0%, #2196F3 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(33,150,243,0.3) !important;
}

.gr-input, .gr-box {
    border-radius: 15px !important;
    border: 2px solid #E3F2FD !important;
    transition: all 0.3s ease !important;
    background: white !important;
}

.gr-input:hover, .gr-box:hover {
    border-color: #2196F3 !important;
    box-shadow: 0 0 15px rgba(33,150,243,0.2) !important;
}

.measurement-box {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 1rem 0;
    animation: slideIn 0.5s ease-out;
    border: 1px solid #E3F2FD;
    transition: all 0.3s ease;
}

.measurement-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.image-container {
    position: relative;
    overflow: hidden;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    background: white;
    padding: 0.5rem;
    border: 1px solid #E3F2FD;
}

.image-container:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.measurement-value {
    font-size: 1.4em;
    font-weight: bold;
    color: #1976D2;
    margin-bottom: 0.5rem;
}

.measurement-label {
    color: #546E7A;
    font-size: 1em;
    font-weight: 500;
}

.app-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.app-footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
"""

def preprocess_image_for_display(img):
    """Ensure the image fits properly in the display box"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    max_size = 800
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width*ratio), int(height*ratio))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    return img

def preprocessing(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (256, 256))
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    img = img/256
    return img

def prediction(img, model):
    img = np.expand_dims(img, 0)
    prediction = (model.predict(img)[0, :, :, :] > 0.1).astype(np.uint8)
    img = (prediction * 255).astype(np.uint8)
    return img

def final_preprocessing(output_img, input_img):
    pixel_to_cm = 0.02
    kernel = np.ones((5, 5), np.uint8)
    
    # Image processing
    blurred = cv2.GaussianBlur(output_img, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresholded, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    img = np.expand_dims(eroded, 2)
    
    # Contour detection
    _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize result images
    input_img = (input_img * 255).astype(np.uint8)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    black_img = np.zeros(input_img.shape, np.uint8)
    
    measurements = {"HC": 0, "BPD": 0, "OFD": 0}
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Calculate measurements
            major_axis = max(axes)
            minor_axis = min(axes)
            a = major_axis / 2.0
            b = minor_axis / 2.0
            
            # Calculate HC using Ramanujan's approximation
            circumference_pixels = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
            
            # Store measurements
            measurements["HC"] = round(circumference_pixels * pixel_to_cm, 2)
            measurements["BPD"] = round(minor_axis * pixel_to_cm, 2)
            measurements["OFD"] = round(major_axis * pixel_to_cm, 2)
            
            # Draw measurements on images
            center = tuple(map(int, center))
            major_axis_vector = (int(a * math.cos(math.radians(angle + 90))), 
                               int(a * math.sin(math.radians(angle + 90))))
            minor_axis_vector = (int(b * math.cos(math.radians(angle))), 
                               int(b * math.sin(math.radians(angle))))
            
            # Draw ellipse and measurements with enhanced colors
            cv2.ellipse(black_img, ellipse, (255, 255, 255), 2)
            cv2.ellipse(input_img, ellipse, (46, 204, 113), 2)
            output_2 = input_img.copy()
            # Draw OFD (major axis)
            cv2.line(output_2, 
                    (center[0] - major_axis_vector[0], center[1] - major_axis_vector[1]),
                    (center[0] + major_axis_vector[0], center[1] + major_axis_vector[1]),
                    (231, 76, 60), 2)
            
            # Draw BPD (minor axis)
            cv2.line(output_2,
                    (center[0] - minor_axis_vector[0], center[1] - minor_axis_vector[1]),
                    (center[0] + minor_axis_vector[0], center[1] + minor_axis_vector[1]),
                    (52, 152, 219), 2)
            
            # # Add measurements text
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(black_img, f"OFD: {measurements['OFD']} cm", (10, 30), font, 0.7, (231, 76, 60), 2)
            # cv2.putText(black_img, f"BPD: {measurements['BPD']} cm", (10, 60), font, 0.7, (52, 152, 219), 2)
            # cv2.putText(black_img, f"HC: {measurements['HC']} cm", (10, 90), font, 0.7, (255, 255, 255), 2)
    
    # Prepare images for display
    input_img = preprocess_image_for_display(input_img)
    output_2 = preprocess_image_for_display(output_2)
    
    return (input_img, output_2, 
            f"HC: {measurements['HC']} cm",
            f"BPD: {measurements['BPD']} cm",
            f"OFD: {measurements['OFD']} cm")

def process_image(input_image):
    if input_image is None:
        return None, None, "No measurement", "No measurement", "No measurement"
    
    try:
        model = load_model("u_net_model_v2.h5")
        processed_img = preprocessing(input_image)
        output_img = prediction(processed_img, model)
        return final_preprocessing(output_img, processed_img)
    except Exception as e:
        return None, None, f"Error: {str(e)}", "Error", "Error"

def measurement_component(value, label):
    return f"""
    <div class="measurement-box">
        <div class="measurement-value">{value}</div>
        <div class="measurement-label">{label}</div>
    </div>
    """

# Create Gradio interface with enhanced layout
with gr.Blocks(css=custom_css) as iface:
    gr.HTML("""
        <div class="app-header">
            <h1 style="color: #1976D2; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700;">
                Fetal Head Measurements Analysis
            </h1>
            <p style="color: #546E7A; font-size: 1.2rem; line-height: 1.6;">
                Upload an ultrasound image to automatically calculate HC, BPD, and OFD measurements using AI-powered analysis
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="filepath",
                label="Upload Ultrasound Image",
                elem_classes="image-container"
            )
            upload_button = gr.Button(
                "Process Image",
                variant="primary"
            )

        with gr.Column(scale=2):
            with gr.Row():
                original_image = gr.Image(
                    type="numpy",
                    label="Original Image with Measurements",
                    elem_classes="image-container"
                )
                processed_image = gr.Image(
                    type="numpy",
                    label="Measurement Visualization",
                    elem_classes="image-container"
                )
            
            with gr.Row():
                with gr.Column():
                    hc_output = gr.Markdown(
                        value=measurement_component("--", "Head Circumference (HC)"),
                        elem_classes="measurement-box"
                    )
                with gr.Column():
                    bpd_output = gr.Markdown(
                        value=measurement_component("--", "Biparietal Diameter (BPD)"),
                        elem_classes="measurement-box"
                    )
                with gr.Column():
                    ofd_output = gr.Markdown(
                        value=measurement_component("--", "Occipitofrontal Diameter (OFD)"),
                        elem_classes="measurement-box"
                    )

    upload_button.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[
            original_image,
            processed_image,
            hc_output,
            bpd_output,
            ofd_output
        ]
    )

    gr.HTML("""
        <div class="app-footer">
            <p style="color: #546E7A; margin: 0;">
                Powered by U-Net Deep Learning Model • Built with Gradio
            </p>
        </div>
    """)

# Launch the interface
iface.launch(share=True)