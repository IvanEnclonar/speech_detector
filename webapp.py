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
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=16000, duration=3)

    # Generate the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Create the plot (no display)
    plt.figure(figsize=(10, 4))

    # Generate the spectrogram without displaying it
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
    url = f'https://drive.google.com/uc?export=download&id=11RNqLLfHJmvjvw7t3AjWk51M5TPGtuHi'

    # Download the file
    gdown.download(url, 'model_weights.pth', quiet=False)

    model = CNN()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    ## Webapp stuff

    st.write(
        """
        # Spectogram Analyzer

        ### DIGIMAP Final Project

        We convert audio files to spectrograms and classify them using a convolutional neural network.
        """
    )

    # Two file input choices
    tab1, tab2 = st.tabs(["Record", "Upload"])
    with tab1:
        uploaded_file = st.file_uploader("Choose a file", type="wav")
    with tab2:
        # There is an issue with st.audio_input. audio_value turns high pitched and loops
        audio_value = st.audio_input("Record a voice message (only the first 3 seconds will be used)")

    if uploaded_file is not None:
        # Read file:
        bytes_data = uploaded_file.getvalue()
        audio_value = BytesIO(bytes_data)
        st.audio(bytes_data)
    
    # Process
    if audio_value:
        st.write(
        """
        ### Results
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
                st.write("### Prediction: Human Speech")
            else:
                st.write("### Prediction: Non-Human Speech")