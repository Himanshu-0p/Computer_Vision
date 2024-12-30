<p class="has-line-data" data-line-start="0" data-line-end="3">Constellation Detection and Analysis<br>
Overview<br>
This project identifies constellations from input images using a pre-trained TensorFlow model, processes the images for graph-based analysis, and displays the results in a graphical user interface (GUI). The program uses machine learning techniques alongside image preprocessing and visualization to achieve its goals.</p>
<p class="has-line-data" data-line-start="4" data-line-end="10">Project Components<br>
<a href="http://one.py">one.py</a>: Main script for running the GUI-based constellation detection application.<br>
<a href="http://preprocess.py">preprocess.py</a>: Module for preprocessing input images to extract features and prepare them for detection and classification.<br>
<a href="http://gnn.py">gnn.py</a>: Script to train and utilize a Graph Neural Network (GNN) model for constellation detection using graph-based representations.<br>
<a href="http://one.py">one.py</a>: Main GUI Application<br>
This script orchestrates the constellation detection pipeline, from loading an image to displaying results.</p>
<p class="has-line-data" data-line-start="11" data-line-end="31">Key Functionalities:<br>
Model Loading: Loads the TensorFlow SavedModel for prediction.<br>
Preprocessing: Resizes and normalizes images for model input.<br>
Prediction: Classifies the input image into one of 30 constellations.<br>
Graph Image Loading: Fetches a corresponding graph representation of the detected constellation.<br>
Class Information: Retrieves detailed information about the detected constellation from a JSON file.<br>
GUI: Implements a user-friendly interface using Tkinter to display the original image, graph, and textual details.<br>
How to Run:<br>
Ensure the required dependencies (cv2, tensorflow, numpy, tkinter, PIL) are installed.<br>
Place the constellation graph images in the Normalised_Templates folder and the constellation information in constellation_info.json.<br>
Run the script:<br>
bash<br>
Copy code<br>
python <a href="http://one.py">one.py</a><br>
Key Files:<br>
Saved Model Directory: saved_model/<br>
Graph Folder: Normalised_Templates/<br>
Constellation Info JSON: constellation_info.json<br>
<a href="http://preprocess.py">preprocess.py</a>: Image Preprocessing Module<br>
This script prepares input images for graph-based constellation detection.</p>
<p class="has-line-data" data-line-start="32" data-line-end="38">Key Functionalities:<br>
Feature Extraction: Extracts keypoints using ORB (Oriented FAST and Rotated BRIEF).<br>
Graph Representation: Converts extracted features into graph structures.<br>
Normalization: Scales features for consistent input to the detection model.<br>
Usage:<br>
Import the ConstellationDetection class in other scripts:</p>
<p class="has-line-data" data-line-start="39" data-line-end="42">python<br>
Copy code<br>
from preprocess import ConstellationDetection</p>
<p class="has-line-data" data-line-start="43" data-line-end="50">constellation_detector = ConstellationDetection(“g”)<br>
constellation_detector.process_image(image_path)<br>
Dependencies:<br>
cv2: For image processing and feature extraction.<br>
numpy: For numerical computations.<br>
<a href="http://gnn.py">gnn.py</a>: Graph Neural Network Model<br>
This script trains and applies a GNN for constellation detection. It uses weighted nodes and multi-head attention mechanisms for better performance.</p>
<p class="has-line-data" data-line-start="51" data-line-end="57">Key Functionalities:<br>
Graph Construction: Builds graphs from star coordinates and connections.<br>
GNN Training: Trains a GNN model to classify constellations based on graph features.<br>
Prediction: Uses the trained model to predict the constellation class from input graphs.<br>
Usage:<br>
Ensure the required libraries (torch, torch_geometric) are installed. Train or load the GNN model and use it as follows:</p>
<p class="has-line-data" data-line-start="58" data-line-end="61">python<br>
Copy code<br>
from gnn import ConstellationGNN</p>
<p class="has-line-data" data-line-start="62" data-line-end="63">model = ConstellationGNN()</p>
<h1 class="code-line" data-line-start=63 data-line-end=64 ><a id="Training_and_prediction_processes_63"></a>Training and prediction processes</h1>
<p class="has-line-data" data-line-start="64" data-line-end="76">Dependencies:<br>
torch: For deep learning functionalities.<br>
torch_geometric: For graph-based learning and operations.<br>
Installation and Setup<br>
Clone the repository and navigate to the project folder.<br>
Install dependencies:<br>
bash<br>
Copy code<br>
pip install -r requirements.txt<br>
Place all necessary assets (model, graph images, JSON) in their respective directories.<br>
Running the Application<br>
Start the GUI application by executing <a href="http://one.py">one.py</a>:</p>
<p class="has-line-data" data-line-start="77" data-line-end="86">bash<br>
Copy code<br>
python <a href="http://one.py">one.py</a><br>
Future Work<br>
Enhance the GNN model for improved accuracy.<br>
Add support for additional constellations.<br>
Enable real-time image capture and detection.<br>
License<br>
This project is open-source and available under the MIT License.</p>
