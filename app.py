import streamlit as st
import numpy as np
from PIL import Image
import io

# Import all necessary Keras applications and preprocessing modules
from tensorflow.keras.applications import (
    ResNet50, ResNet50V2, VGG16, VGG19, Xception, InceptionV3, MobileNetV2,
    DenseNet121, NASNetLarge, EfficientNetV2B0
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input, decode_predictions as resnet50_decode_predictions
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet50v2_preprocess_input, decode_predictions as resnet50v2_decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input, decode_predictions as vgg16_decode_predictions
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input, decode_predictions as vgg19_decode_predictions
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input, decode_predictions as xception_decode_predictions
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input, decode_predictions as inceptionv3_decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input, decode_predictions as mobilenetv2_decode_predictions
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input, decode_predictions as densenet_decode_predictions
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess_input, decode_predictions as nasnet_decode_predictions
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess_input, decode_predictions as efficientnetv2_decode_predictions

from tensorflow.keras.preprocessing import image as keras_image_preprocessing # Alias to avoid conflict with PIL.Image

# Model dictionary containing model constructors, preprocessing functions, decode functions, and target image size
MODEL_INFO = {
    "ResNet50": {
        "model": ResNet50,
        "preprocess": resnet50_preprocess_input,
        "decode": resnet50_decode_predictions,
        "target_size": (224, 224)
    },
    "ResNet50V2": {
        "model": ResNet50V2,
        "preprocess": resnet50v2_preprocess_input,
        "decode": resnet50v2_decode_predictions,
        "target_size": (224, 224)
    },
    "VGG16": {
        "model": VGG16,
        "preprocess": vgg16_preprocess_input,
        "decode": vgg16_decode_predictions,
        "target_size": (224, 224)
    },
    "VGG19": {
        "model": VGG19,
        "preprocess": vgg19_preprocess_input,
        "decode": vgg19_decode_predictions,
        "target_size": (224, 224)
    },
    "Xception": {
        "model": Xception,
        "preprocess": xception_preprocess_input,
        "decode": xception_decode_predictions,
        "target_size": (299, 299) # Xception requires (299, 299)
    },
    "InceptionV3": {
        "model": InceptionV3,
        "preprocess": inceptionv3_preprocess_input,
        "decode": inceptionv3_decode_predictions,
        "target_size": (299, 299) # InceptionV3 requires (299, 299)
    },
    "MobileNetV2": {
        "model": MobileNetV2,
        "preprocess": mobilenetv2_preprocess_input,
        "decode": mobilenetv2_decode_predictions,
        "target_size": (224, 224)
    },
    "DenseNet121": {
        "model": DenseNet121,
        "preprocess": densenet_preprocess_input,
        "decode": densenet_decode_predictions,
        "target_size": (224, 224)
    },
    "NASNetLarge": {
        "model": NASNetLarge,
        "preprocess": nasnet_preprocess_input,
        "decode": nasnet_decode_predictions,
        "target_size": (331, 331) # NASNetLarge requires (331, 331)
    },
    "EfficientNetV2B0": {
        "model": EfficientNetV2B0,
        "preprocess": efficientnetv2_preprocess_input,
        "decode": efficientnetv2_decode_predictions,
        "target_size": (224, 224)
    }
}

# Load the pre-trained model using st.cache_resource for efficiency
@st.cache_resource
def load_model(model_name):
    st.write(f"Loading {model_name}...")
    ModelClass = MODEL_INFO[model_name]["model"]
    model = ModelClass(weights='imagenet')
    st.write(f"{model_name} loaded successfully.")
    return model

st.title('Image Classification with Keras Pre-trained Models')
st.write('Upload an image and select a model to see its predictions!')

# Sidebar for model selection
st.sidebar.title("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a pre-trained model:",
    list(MODEL_INFO.keys()),
    index=0 # Default to ResNet50
)

model = load_model(selected_model_name)
model_info = MODEL_INFO[selected_model_name]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img_bytes = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the selected model
    target_size = model_info["target_size"]
    img_resized = img.resize(target_size)
    img_array = keras_image_preprocessing.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = model_info["preprocess"](img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = model_info["decode"](predictions, top=5)[0] # Get top 5 predictions

    st.subheader(f"Predictions using {selected_model_name}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}: {label} ({score*100:.2f}%) - ImageNet ID: {imagenet_id}")
else:
    st.info("Please upload an image to get predictions.")
