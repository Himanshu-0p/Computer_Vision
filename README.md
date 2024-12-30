README: Constellation Detection and Analysis
Overview
This project identifies constellations from input images using a pre-trained TensorFlow model, processes the images for graph-based analysis, and displays the results in a graphical user interface (GUI). The program uses machine learning techniques alongside image preprocessing and visualization to achieve its goals.

Project Components
one.py: Main script for running the GUI-based constellation detection application.
preprocess.py: Module for preprocessing input images to extract features and prepare them for detection and classification.
gnn.py: Script to train and utilize a Graph Neural Network (GNN) model for constellation detection using graph-based representations.
one.py: Main GUI Application
This script orchestrates the constellation detection pipeline, from loading an image to displaying results.

Key Functionalities:
Model Loading: Loads the TensorFlow SavedModel for prediction.
Preprocessing: Resizes and normalizes images for model input.
Prediction: Classifies the input image into one of 30 constellations.
Graph Image Loading: Fetches a corresponding graph representation of the detected constellation.
Class Information: Retrieves detailed information about the detected constellation from a JSON file.
GUI: Implements a user-friendly interface using Tkinter to display the original image, graph, and textual details.
How to Run:
Ensure the required dependencies (cv2, tensorflow, numpy, tkinter, PIL) are installed.
Place the constellation graph images in the Normalised_Templates folder and the constellation information in constellation_info.json.
Run the script:
bash
Copy code
python one.py
Key Files:
Saved Model Directory: saved_model/
Graph Folder: Normalised_Templates/
Constellation Info JSON: constellation_info.json
preprocess.py: Image Preprocessing Module
This script prepares input images for graph-based constellation detection.

Key Functionalities:
Feature Extraction: Extracts keypoints using ORB (Oriented FAST and Rotated BRIEF).
Graph Representation: Converts extracted features into graph structures.
Normalization: Scales features for consistent input to the detection model.
Usage:
Import the ConstellationDetection class in other scripts:

python
Copy code
from preprocess import ConstellationDetection

constellation_detector = ConstellationDetection("g")
constellation_detector.process_image(image_path)
Dependencies:
cv2: For image processing and feature extraction.
numpy: For numerical computations.
gnn.py: Graph Neural Network Model
This script trains and applies a GNN for constellation detection. It uses weighted nodes and multi-head attention mechanisms for better performance.

Key Functionalities:
Graph Construction: Builds graphs from star coordinates and connections.
GNN Training: Trains a GNN model to classify constellations based on graph features.
Prediction: Uses the trained model to predict the constellation class from input graphs.
Usage:
Ensure the required libraries (torch, torch_geometric) are installed. Train or load the GNN model and use it as follows:

python
Copy code
from gnn import ConstellationGNN

model = ConstellationGNN()
# Training and prediction processes
Dependencies:
torch: For deep learning functionalities.
torch_geometric: For graph-based learning and operations.
Installation and Setup
Clone the repository and navigate to the project folder.
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Place all necessary assets (model, graph images, JSON) in their respective directories.
Running the Application
Start the GUI application by executing one.py:

bash
Copy code
python one.py
Future Work
Enhance the GNN model for improved accuracy.
Add support for additional constellations.
Enable real-time image capture and detection.
License
This project is open-source and available under the MIT License.
