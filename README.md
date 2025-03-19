
# **ParkinsonNet**  
**A Convolutional Neural Networks Model for Parkinson’s Disease Detection from Images and Voice Data**  

## **Overview**  
ParkinsonNet is a deep learning-based application that detects Parkinson’s disease using images and voice data. The system utilizes Convolutional Neural Networks (CNNs) for classification, allowing users to upload MRI images or voice samples for prediction. The application features an intuitive GUI built with Tkinter and provides real-time results.  

## **Key Features**  
- **Image-Based Detection**: Classifies Parkinson’s disease from medical images.  
- **Voice-Based Detection**: Analyzes voice data to predict the presence of Parkinson’s.  
- **Machine Learning Model Integration**: Uses pre-trained CNN models for high accuracy.  
- **Graphical Representation**: Displays model performance graphs.  
- **User-Friendly Interface**: Simple Tkinter-based GUI for easy interaction.  

## **Challenges**  
- **Data Accuracy**: Ensuring the models generalize well across different datasets.  
- **Model Optimization**: Improving efficiency for real-time processing.  

## **Future Enhancements**  
- **Improved NLP for Voice Analysis**: Enhance voice recognition for better accuracy.  
- **Real-Time Data Processing**: Enable live detection from microphones.  
- **Cloud Integration**: Store and process data using cloud-based services.  

## **Python Version**  
- Python 3.11.9 or higher  

## **Steps to Start the Project**  
1. **Setup Python Environment**: Install Python 3.11.9 or later.  
2. **Install Required Libraries**: Use the installation commands provided below.  
3. **Run the Program**: Execute the `ParkinsonPrediction.py` script to launch the GUI.  
4. **Upload Image/Voice Data**: Select a file and run predictions.  

## **Libraries and Installation Commands**  

1. **Tkinter (GUI)**  
   - **Install Command**: (Pre-installed with Python)  
   - **Use**: For creating a graphical user interface.  

2. **OpenCV**  
   - **Install Command**:  
     ```bash
     pip install opencv-python
     ```  
   - **Use**: Image processing and resizing.  

3. **NumPy**  
   - **Install Command**:  
     ```bash
     pip install numpy
     ```  
   - **Use**: Handling array-based data.  

4. **Pandas**  
   - **Install Command**:  
     ```bash
     pip install pandas
     ```  
   - **Use**: Reading and processing CSV files.  

5. **Matplotlib**  
   - **Install Command**:  
     ```bash
     pip install matplotlib
     ```  
   - **Use**: Plotting accuracy and loss graphs.  

6. **Keras & TensorFlow**  
   - **Install Commands**:  
     ```bash
     pip install keras tensorflow
     ```  
   - **Use**: Loading and running deep learning models.  

7. **Pillow**  
   - **Install Command**:  
     ```bash
     pip install pillow
     ```  
   - **Use**: Handling image formats in Tkinter.  

8. **Pickle**  
   - **Install Command**:  
     ```bash
     pip install pickle-mixin
     ```  
   - **Use**: Loading pre-trained model histories.  

## **Usage**  

### **1. Image-Based Detection**  
- Click **"Detect Parkinson from Images"**  
- Select an MRI or medical image.  
- The model will classify it as **Healthy** or **Parkinson’s**.  

### **2. Voice-Based Detection**  
- Click **"Detect Parkinson from Voice Samples"**  
- Select a CSV file containing voice data.  
- The model will classify it as **Healthy** or **Parkinson’s**.  

### **3. View Model Performance**  
- Click **"Machine Learning Performance Graph"**  
- View accuracy and loss graphs for both image and voice models.  

### **4. Exit Application**  
- Click **"Exit"** to close the program.  

## **Project Structure**  
