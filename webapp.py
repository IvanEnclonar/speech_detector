import streamlit as st

## For sound to image conversion
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.data import DataLoader
import gdown

## CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Second convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 2)  # Output layer (2 classes: human speech, non-human speech)

    def forward(self, x):
        # Apply convolutional layers followed by ReLU activation and max pooling
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))

        # Flatten the tensor before passing it to the fully connected layers
        x = x.view(-1, 64 * 32 * 32)

        # Fully connected layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

        return x

def convertAudioToImage(audio_file):
    # Load the audio file with a sampling rate of 16 kHz and a duration of 3 seconds. The sampling rate is set to 16 kHz to match the model's input size and the duration is limited to 3 seconds to reduce the processing time.
    # The sampling rate of an audio file is the number of samples of audio carried per second, measured in Hz or kHz. A higher sampling rate captures more details of the audio signal but requires more storage space and processing power.
    y, sr = librosa.load(audio_file, sr=16000, duration=3)

    # Generate the mel spectrogram
    # A Mel spectrogram is a visual representation of the frequency content of an audio signal over time, mapped onto the Mel scale, which reflects human auditory perception. It emphasizes lower frequencies where humans are more sensitive and compresses higher frequencies. The x-axis represents time, the y-axis shows frequencies in the Mel scale, and the color intensity indicates the loudness of frequencies at each time point. This makes it useful for analyzing speech, music, and sounds in a way that aligns with human hearing.
    # The y parameter is the audio signal, sr is the sampling rate, n_mels is the number of Mel bands to generate, and fmax is the maximum frequency to include in the spectrogram. 
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to log scale (dB). This is a common practice in audio processing to compress the dynamic range of the spectrogram and make it easier to visualize and analyze.
    S_db = librosa.power_to_db(S, ref=np.max)

    # Create the plot (no display)
    plt.figure(figsize=(10, 4))

    # Generate the spectrogram using librosa's display module.
    librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr)

    plt.axis('off')  # Hide the axis labels
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove extra space around the plot

    # Save the spectrogram to a BytesIO object
    image_bytes = BytesIO()
    plt.savefig(image_bytes, format='png', bbox_inches='tight', pad_inches=0)
    image_bytes.seek(0)  # Rewind to the beginning of the BytesIO object

    # Close the plot to free memory
    plt.close()

    del y, S, S_db

    return image_bytes

from torchvision import transforms
from PIL import Image
import torch

def convertImageToTensor(image_bytes):
    # Convert BytesIO to PIL Image
    image = Image.open(image_bytes).convert("RGB")  # Ensure 3 channels (RGB)

    # from google colab file
    transform = transforms.Compose([
        transforms.Resize((128, 128)),        # Resize images to a fixed size
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Apply the transformations
    tensor = transform(image)

    return tensor



if __name__ == '__main__':
    ## Load the CNN model
    # URL or file ID
    url = f'https://drive.google.com/uc?export=download&id=1TkxXAN8QTZcQwHWYfH2oqI3sWK0Aqa3t'

    # Download the file
    gdown.download(url, 'model_weights.pth', quiet=False)

    model = CNN()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    ## Webapp stuff

    st.write(
        """
        # Spectogram Analyzer

        #### DIGIMAP Final Project

        Our Speech Detector project leverages an innovative approach to detect speech by analyzing spectrograms instead of raw audio. Using a Convolutional Neural Network (CNN) trained from scratch on the LibriSpeech and UrbanSound8K datasets, the model processes spectrograms to determine whether a person is speaking or not. The system transforms audio input into a spectrogram, which is then fed into the CNN for classification. This allows the model to accurately distinguish between human speech and non-human sounds. The project is implemented in Python and deployed through a web application, providing an easy-to-use interface for real-time speech detection.
        """
    )

    # Two file input choices
    tab1, tab2, tab3 = st.tabs(["Record", "Upload", "How Kernels Work"])
    with tab1:
        audio_value = st.audio_input("Record a voice message (only the first 3 seconds will be used)")
    with tab2:
        # There is an issue with st.audio_input. audio_value turns high pitched and loops
        uploaded_file = st.file_uploader("Choose a file", type="wav")
    with tab3:
        st.write(
            """
            ## How Kernels Work
            
            Kernels (or filters) are small matrices learned during the training process of the model. These Kernels slide over the input image, extracting specific patterns or shapes that help the model distinguish human speech within a spectrogram image. The 3 images below are the first kernel for the 3 channels of the images of our model. Combined with other kernels, they help the model learn to detect human speech.
            """
        )
        st.image("Screenshot 2024-12-06 235011.png", caption="Kernel 1", width=300)
        
        st.write("When applied to this image of a human speech spectrogram:")
        st.write("human.png", caption="Human Speech Spectrogram", width=300)
        st.write("The output image is:")
        st.image("Screenshot 2024-12-06 235128.png", caption="Kernel 1 applied to an image", width=300)
        
        st.write("The resulting image of other kernels applied to the spectogram can be seen below:")
        st.image("Total.png", caption="Kernels applied to an image", width=300)
        
    if uploaded_file is not None:
        # Read file:
        bytes_data = uploaded_file.getvalue()
        audio_value = BytesIO(bytes_data)
        st.audio(bytes_data)
    
    # Process
    if audio_value:
        st.write(
        """
        ## Results
        """
        )

        input_image = convertAudioToImage(audio_value)

        st.write("#### Spectogram")
        st.image(input_image)

        with torch.no_grad():
            # Forward pass
            tensor = convertImageToTensor(input_image)
            outputs = model(tensor)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            if predicted.item() == 0:
                st.write("#### Prediction: Human Speech")
            else:
                st.write("#### Prediction: Non-Human Speech")