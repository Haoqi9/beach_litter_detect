import base64
import cv2
import numpy as np
import streamlit as st
import glob
import tempfile
from ultralytics import YOLO

################################################################################

# PAGE CONFIGURATION

st.set_page_config(
    page_icon='logo.jpg',
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
    image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
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

RGB_dict_yolo = {
    'cardboard' : (10, 41, 252),
    'plastic'   : (12, 219, 237),
    'glass'     : (244, 244, 239),
    'metal'     : (0, 226, 179),
    'text'      : (18, 32, 100),
    'white'     : (255, 255, 255)
}

RGB_dict_yolo_reverse = {
    'cardboard' : (252, 41, 10),
    'plastic'   : (237, 219, 12),
    'glass'     : (239, 244, 244),
    'metal'     : (179, 226, 0),
    'text'      : (100, 32, 18),
    'white'     : (255, 255, 255)
}

################################################################################

# BODY

# Convert the local image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Path to your local image
image_path = "logo.jpg"
data_url = image_to_base64(image_path)

st.markdown(f"""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 60vh;">
        <img src="data:image/jpeg;base64,{data_url}" alt="Logo" style="width: 230px;">
        <h1 style="margin-left: 40px;">
            <span style="color: #2B5D65;">Trash</span><span style="color: #47BFB6;">Tracker</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

#1E90FF (blue)
#90EE90

# title1, title2 = st.columns([0.4, 0.6])
# title1.image('logo.jpg', width=250)
# title2.markdown(
#     """
#     <div style="display: flex; justify-content: flex-start; align-items: center; height: 25vh; padding-left: 5px;">
#         <h1 style="color: #2E8B57;">TrashTracker</h1>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# st.write(
#     '<h1 style="text-align: center;">🗑️♻️ Detector de Basura en Playas ☀️🏖️</h1>', unsafe_allow_html=True
# )

tab1, tab2 = st.tabs(["Images", "Videos"])
# Annotate Images.
with tab1: 
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
            param1, param2 = st.columns([0.5, 0.5], gap='large')
            param3, param4 = st.columns([0.5, 0.5], gap='large')
            
            conf_text = "Controls which predictions are kept based on their confidence score (i.e., how confident the model is that the box contains an object). If a bounding box's confidence score is below the conf threshold, it will be discarded."

            iou_text = "Controls the overlap threshold used during Non-Maximum Suppression to remove duplicate predictions (i.e., how much overlap between boxes is allowed before one is discarded). If the IoU is below the threshold, the boxes will be considered as separate detections."

            conf_threshold = param1.number_input(
                label='**Confidence Score Threshold**:',
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help=conf_text
            )

            iou_threshold = param3.number_input(
                label='**IoU Threshold**:',
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help=conf_text
            )

            full_screen = param2.radio(
                label='**Show image in full screen**:',
                options=['Yes', 'No'],
                index=0
            )
            
            pred_style = param4.radio(
                label='**Choose prediction style**:',
                options=[
                    '*YOLO default*: pred boxes + labels',
                    '*Custom* : pred boxes + nº instance'
                ],
                index=0
            )

            # submit button.
            submitted = st.form_submit_button(label='Predict')
        
        if submitted:
            img_np = get_image_np_from_bytes(uploaded_img)

            pred_results = model.predict(img_np,
                                        conf=conf_threshold,
                                        iou=iou_threshold
                                        )[0]

            if full_screen == 'Yes':
                use_column_width=True
            else:
                use_column_width=False

            # Display annotated image.
            st.write(
                '<h2 style="text-align: center;">Predicted Image</h2>', unsafe_allow_html=True
            )

            if pred_style == '*Custom* : pred boxes + nº instance':
                # make the array compatible with cv2.
                pred_img = np.ascontiguousarray(img_np, dtype=np.uint8)
            else:
                pred_img = pred_results.plot(conf=False)
            
            # Count class detected.
            detect_class_count_dict = {clase:0 for clase in class_list}
            for i, detect_img_box in enumerate(pred_results.boxes):
                detect_class = class_list[int(detect_img_box.cls)]
                detect_class_count_dict[detect_class] += 1

                if pred_style == '*Custom* : pred boxes + nº instance':
                    x1, y1, x2, y2 = map(int, detect_img_box.xyxy[0])
                    # Draw bounding box.
                    pred_img = cv2.rectangle(pred_img, (x1, y1), (x2, y2), RGB_dict_yolo_reverse[detect_class], 2)
                    # Put instance number.
                    pred_img = cv2.putText(pred_img, str(i+1), (x1 + 15, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    # Put class label.
                    # pred_img = cv2.putText(pred_img, detect_class, (x1 + 15, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

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
                    if clase == class_list[0]:
                        text_RGB = RGB_dict_yolo['white']
                    else:
                        text_RGB = RGB_dict_yolo['text']
                        
                    detec_containers[i].write(f"""<p style="background-color:rgb{RGB_dict_yolo[clase]}; color:rgb{text_RGB}; text-align:center;"><b>{clase.capitalize()} ({count})</b></p>""",
                    unsafe_allow_html=True)

            st.image(
                image=pred_img[..., ::-1],
                use_column_width=use_column_width
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

                        # resize image.
                        # cropped_img = cv2.resize(cropped_img, dsize=(800, 800), interpolation=cv2.INTER_LINEAR)

                        detection_counter += 1
                        
                        if class_name == class_list[0]:
                            text_RGB = RGB_dict_yolo['white']
                        else:
                            text_RGB = RGB_dict_yolo['text']
                            
                        container.write(f"""<p style="background-color:rgb{RGB_dict_yolo[class_name]}; color:rgb{text_RGB}"><b>Instance {detection_counter}: {class_name.capitalize()} ({conf_score})</b></p>""", unsafe_allow_html=True)
                        container.image(image=cropped_img[..., ::-1], use_column_width=True)
        
with tab2:
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        video_placeholder = st.empty()

        # Create a temporary file to save the video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name
        
        # Open the video using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        
        # Loop through video frames
        unique_ids = set()
        detect_class_count_dict = {clase:0 for clase in class_list}
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Perform YOLO prediction on the frame
            results = next(model.track(frame,
                                  persist=True,
                                  stream=True,
                                  conf=0.25,
                                  iou=0.5
                                  ))

            labeled_frame = results.plot(conf=False)
            
            for detect_img_box in results.boxes:
                # DEBUG.
                # print(detect_img_box)
                if detect_img_box.is_track:
                    detect_id = int(detect_img_box.id)
                    if detect_id not in unique_ids:
                        detect_class = class_list[int(detect_img_box.cls)]
                        detect_class_count_dict[detect_class] += 1
                        unique_ids.add(detect_id)
            
            # Definir texto de conteo.
            total_count = str(len(unique_ids))
            text = 'Total count:' + total_count
            tuplas_clase_count = [(clase.capitalize(), str(count)) for clase, count in detect_class_count_dict.items()]
            for tupla in tuplas_clase_count:
                text += f" | {':'.join(tupla)}"
            
            # # Añadir texto en imagen anotado.
            # labeled_frame = put_text_in_img_middle_upper(
            #     image_np=labeled_frame,
            #     text=text,
            #     font_scale=0.6,
            #     font_thickness=2,
            #     color=(255, 255, 255)  # white.
            # )
            
            video_placeholder.image(
                image=labeled_frame,
                caption=f'{text}',
                channels="BGR")
