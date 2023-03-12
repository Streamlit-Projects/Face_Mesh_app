import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

import cv2
import mediapipe as mp
from PIL import Image

import tempfile
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo.jpg'

st.title('MediaPipe - Face Mesh')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expended="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expended="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,
)

st.sidebar.title('Face Mesh - app')


@st.cache()
def image_resize(image, width = None, height = None, inter_=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    # Resise the image:
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


#app_mode = st.sidebar.selectbox('Choose the App mode:'
#                               ,['About App', 'Run on Image'])

with st.sidebar:
    selected = option_menu("Menu", ["About", 'Run on Image'],
        icons=['info-square', 'image'], default_index=0)

if selected == 'About':
    st.markdown("MediaPipe Face Mesh is a computer vision technology developed by Google's MediaPipe team. "
                "It is a machine learning model that can accurately detect and track facial landmarks in real-time.")

    st.markdown(
        "Facial landmarks refer to specific points on the face, such as the corners of the eyes, the tip of the nose,"
        " and the edges of the mouth. These landmarks can be used for a variety of applications, including facial recognition,"
        " augmented reality, and emotion detection.")

    st.markdown("MediaPipe Face Mesh uses a deep neural network to predict the 3D positions of 468 facial landmarks."
                " This model can be run on a variety of devices, including smartphones and laptops, making it highly accessible.")

    st.markdown(
        "The technology has been used in a variety of applications, such as virtual try-on experiences, interactive "
        "filters for social media, and real-time facial animation.")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expended="true"] > div:first-child{
            width: 350px
            }
        [data-testid="stSidebar"][aria-expended="false"] > div:first-child{
            width: 350px
            margin-left: -350px
            }
        </style>
        """,
        unsafe_allow_html = True,
            )


############################################################ IMAGE PART #######################################################################

elif selected == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness = 2, circle_radius = 1)
    st.sidebar.markdown('---')
    st.sidebar.subheader('Parameters:')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expended="true"] > div:first-child{
            width: 350px
            }
        [data-testid="stSidebar"][aria-expended="false"] > div:first-child{
            width: 350px
            margin-left: -350px
            }
        </style>
        """,
        unsafe_allow_html = True,
            )

    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")

    max_faces = st.sidebar.number_input("Maximum Number of Face(s)", value = 2, min_value=1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider("Min Detection Confidence", min_value = 0.0, max_value =1.0, value = 0.5)
    st.sidebar.markdown("---")

    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type = ["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)


    face_count = 0


    # To improve performance, optionally mark the image as not writeable to pass by reference:
    image = cv2.cvtColor(
        cv2.flip(image, 1),
        cv2.COLOR_BGR2RGB)
    image.flags.writeable = False


    ## DASHBOARD PART
    with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = max_faces,
    min_detection_confidence = detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        # Face landmarkes Drawing:
        for face_landmarks in results.multi_face_landmarks:
            face_count +=1

            mp_drawing.draw_landmarks(
            image = out_image,
            landmark_list = face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

            kpi1_text.write(f"<h1 styles='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html = True)


        st.subheader('Output Image')
        st.image(out_image, use_column_width = True)


