# Speech Detector  

This repository contains a speech detection project developed as part of our DIGIMAP class. The project takes an innovative approach by using spectrograms—visual representations of audio signals—instead of raw audio data to determine if a person is speaking. The spectrograms are processed by a Convolutional Neural Network (CNN) built and trained using PyTorch.  

The model was trained from scratch using two open-source datasets:  
- [LibriSpeech](https://www.openslr.org/12)  
- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)  

### **Web Application**  
You can access the web application showcasing our speech detector here:  
[Speech Detector Web App](https://ivanenclonar-speech-detector-webapp-prod-wmjda9.streamlit.app/?fbclid=IwZXh0bgNhZW0CMTEAAR2dhs36V3ceas7pK5A_-rnm0Qa9wTWqWPmdteuaOEk25t1AQBQvvPQmxV4_aem_bIPKpmJkI7JTMFgXUIhdhA)  

---

## **Getting Started**

### **Prerequisites**
Ensure you have Python installed and the following dependencies:  
- `streamlit`  
- `librosa`  
- `numpy`  
- `torch`  
- `torchvision`  
- `torchaudio`  
- `gdown`  
- `matplotlib`  

Install the dependencies using the following command:  
```bash
pip install streamlit librosa numpy torch torchvision torchaudio gdown matplotlib
```

---

### **How to Run the Program**

1. **Train the Model**  
   Use the provided `CNN.ipynb` notebook to train the speech detection model.  

2. **Run the Web Application**  
   Once the model is trained, use the trained model in the web app. Run the following command to start the app using Streamlit:  
   ```bash
   streamlit run webapp.py
   ```  

---

## **How It Works**
1. Converts audio input into spectrograms for processing.  
2. Passes the spectrogram through a CNN to classify whether a person is speaking.  
3. Outputs results through a user-friendly web interface.  

---

## **License**  
This project is licensed under the [MIT License](LICENSE), allowing you to freely use and share the code.  

---

Feel free to fork, contribute, or use this repository as inspiration for your own projects. If you have any questions or feedback, let us know!  
