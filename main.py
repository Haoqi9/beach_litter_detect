import cv2
import numpy as np
import streamlit as st
import glob
from ultralytics import YOLO

################################################################################

# PAGE CONFIGURATION

st.set_page_config(
    page_title='Litter Detector',
    layout='wide',
    initial_sidebar_state='auto',
)

################################################################################

# FUNCTIONS

def get_image_np_from_bytes (img_bytesIO):
    # Read the binary data in from image file and transform to bytearray to make it mutable.
    img_bytes = bytearray(img_bytesIO.read())
    # Convert bytearray to numpy array (1 dimension).
    np_array = np.asarray(img_bytes, dtype=np.uint8)
    # Decode numpy array to image tensor (3 dimensions).
    image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)[..., ::-1]
    return image_np

def put_text_in_img_middle_upper(
    image_np,
    text,
    font_scale=0.8,
    font_thickness=2,
    color=(255, 255, 255)  # white.
):
    # Load image tensor (np.array).
    image = image_np
    
    # Get image dimensions.
    height, width, _ = image.shape
    
    # Define the text and font.
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate the size of the text.
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Calculate X position to center the text.
    x = (width - text_width) // 2
    
    # Calculate Y position (slightly below the top of the image).
    y = int(height * 0.05) + text_height  # Adjust the 0.1 multiplier to move the text up or down.
    
    # Add text to the image.
    mod_image_tensor = cv2.putText(image, text, (x, y), font, font_scale, color, font_thickness)
    return mod_image_tensor
    
################################################################################

# LOADING EXTERNAL RESOURCES

# import trained YOLOv8 model(s).
@st.cache_data
def load_model(model_path):
    model = YOLO(model_path, task='detect')
    return model


# allows you to search for files that match a specific pattern. Here, *.pt is the pattern used.
pt_files = glob.glob("*.pt")

if len(pt_files) == 1:
    model_name = pt_files[0]
    mode_path = model_name
    model = load_model(mode_path)
else:
    raise Exception(f"Found {len(pt_files)} .pt files: {pt_files}!")

class_list = ['cardboard', 'plastic', 'glass', 'metal']

color_dict = {
    'cardboard' : 'blue',
    'plastic'   : 'orange',
    'glass'     : 'green',
    'metal'     : 'red'
}

RGB_dict = {
    'blue'   : (0, 0, 255),
    'orange' : (255, 165, 0),
    'green'  : (0, 255, 0),
    'black'  : (0, 0, 0),
    'red'    : (255, 0, 0)
}

################################################################################

# BODY

st.write(
    '<h1 style="text-align: center;">🗑️♻️ Detector de Basura en Playas ☀️🏖️</h1>', unsafe_allow_html=True
)

# Annotate Images.
# Yolo model used.
st.caption(f"Model: {model_name}")

# Image uploader widget for images.
uploaded_img = st.file_uploader(
    "Upload an image file ...",
    type=["jpg", "jpeg", "png"],
    key='images'
)


if uploaded_img is not None:
    # Display parameters.
    st.write(
        '<h2 style="text-align: center;">Parameters</h2>', unsafe_allow_html=True
    )

    with st.form(key='form1'):
        param1, param2, param3 = st.columns(3)
        
        conf_text = "Controls which predictions are kept based on their confidence score (i.e., how confident the model is that the box contains an object). If a bounding box's confidence score is below the conf threshold, it will be discarded."

        iou_text = "Controls the overlap threshold used during Non-Maximum Suppression to remove duplicate predictions (i.e., how much overlap between boxes is allowed before one is discarded). If the IoU is below the threshold, the boxes will be considered as separate detections."

        conf_threshold = param1.number_input(
            label='Confidence Score Threshold',
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help=conf_text
        )

        iou_threshold = param2.number_input(
            label='IoU Threshold',
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help=conf_text
        )

        full_screen = param3.toggle(
            label='See full screen',
            value=True
        )

        # submit button.
        submitted = st.form_submit_button()
    
    if submitted:
        img_np = get_image_np_from_bytes(uploaded_img)

        pred_results = model.predict(img_np,
                                    conf=conf_threshold,
                                    iou=iou_threshold
                                    )[0]

        if full_screen:
            use_column_width=True
        else:
            use_column_width=False

        # Display annotated image.
        st.write(
            '<h2 style="text-align: center;">Predicted Image</h2>', unsafe_allow_html=True
        )

        # make the array compatible with cv2.
        pred_img = np.ascontiguousarray(img_np, dtype=np.uint8)
        # Count class detected.
        detect_class_count_dict = {clase:0 for clase in class_list}
        for detect_img_box in pred_results.boxes:
            detect_class = class_list[int(detect_img_box.cls)]
            detect_class_count_dict[detect_class] += 1
            class_color = color_dict[detect_class]

            x1, y1, x2, y2 = map(int, detect_img_box.xyxy[0])
            # Draw bounding box.
            pred_img = cv2.rectangle(pred_img, (x1, y1), (x2, y2), RGB_dict[class_color], 2)
            # Put class label.
            # pred_img = cv2.putText(pred_img, detect_class, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, RGB_dict[class_color], 1)

        # Display detection count info.
        tuplas_clase_count = [(clase, str(count)) for clase, count in detect_class_count_dict.items()]
        total_count = sum([int(count) for _, count in tuplas_clase_count])

        st.write(f"""<p style="text-align:center; background-color:black; color:white">Detections ({total_count})</p>""",
        unsafe_allow_html=True)

        # Do not run if 0 detection.
        if total_count > 0:
            tuplas_clase_pos_count = [tupla for tupla in tuplas_clase_count if int(tupla[1]) > 0]
            detec_containers = [f"detec_container{i}" for i in range(len(tuplas_clase_pos_count)+1)]
            detec_containers = st.columns(len(tuplas_clase_pos_count))
            for i, (clase, count) in enumerate(tuplas_clase_pos_count):
                detec_containers[i].write(f"""<p style="background-color:{color_dict[clase]}; color:white; text-align:center;">{clase.capitalize()} ({count})</p>""",
                unsafe_allow_html=True)

        st.image(
            # pred_img,
            pred_img,
            use_column_width=use_column_width
        )

        # Encode np array image back to binary format to be saved.
        success, encoded_image = cv2.imencode('.png', pred_img[..., ::-1])
        if not success:
            raise RuntimeError("Failed to encode image.")

        # Convert encoded image to binary format
        binary_image = encoded_image.tobytes()

        st.download_button(
            label='Download annotated image',
            data=binary_image,
            file_name='annotated_image.png',
            key=3
        )
        
        # Display individual detections.
        st.write(
            '<h2 style="text-align:center;">Individual Detections</h2>', unsafe_allow_html=True
        )
        
        # Do not run if 0 detection.
        if total_count > 0:
            # Create containers for each ind detection.
            cols_per_row = 5
            containers_list = [f"container{i}" for i in range(len(pred_results)+1)]
            pred_boxes = pred_results.boxes
            detection_counter = 0
            for i in range(0, len(pred_results)+1, cols_per_row):
                pred_boxes_chunk = pred_boxes[i:i+cols_per_row]
                containers_chunk = containers_list[i:i+cols_per_row]
                containers_chunk = st.columns(cols_per_row)
            
                for i, (pred_box, container) in enumerate(zip(pred_boxes_chunk, containers_chunk)):
                    class_id = int(pred_box.cls[0])
                    class_name = class_list[class_id]
                
                    conf_score = np.round(float(pred_box.conf[0]), 2)
                    
                    detection_xyxy = pred_box.xyxy
                    x_min, y_min, x_max, y_max = detection_xyxy[0]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    # new orig_img so do not overwrite bouding boxes in same img.
                    cropped_img = img_np[y_min:y_max, x_min:x_max]

                    detection_counter += 1
                    container.write(f"""<p style="background-color:{color_dict[class_name]}; color:white">Instance {detection_counter}: {class_name.capitalize()} ({conf_score})</p>""", unsafe_allow_html=True)
                    container.image(image=cropped_img, use_column_width=True)
