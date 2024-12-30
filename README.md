<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Constellation Detection and Analysis</title>
</head>
<body>
    <h1>Constellation Detection and Analysis</h1>

    <h2>Overview</h2>
    <p>
        This project identifies constellations from input images using a pre-trained TensorFlow model, processes the images for graph-based analysis, and displays the results in a graphical user interface (GUI). The program combines machine learning techniques with image preprocessing and visualization.
    </p>

    <hr>

    <h2>Project Components</h2>
    <ul>
        <li><strong><code>one.py</code></strong>: Main script for running the GUI-based constellation detection application.</li>
        <li><strong><code>preprocess.py</code></strong>: Module for preprocessing input images to extract features and prepare them for detection and classification.</li>
        <li><strong><code>gnn.py</code></strong>: Script to train and utilize a Graph Neural Network (GNN) model for constellation detection using graph-based representations.</li>
    </ul>

    <h3><code>one.py</code>: Main GUI Application</h3>
    <p>This script orchestrates the constellation detection pipeline, from loading an image to displaying results.</p>

    <h4>Key Functionalities:</h4>
    <ul>
        <li><strong>Model Loading</strong>: Loads the TensorFlow SavedModel for prediction.</li>
        <li><strong>Preprocessing</strong>: Resizes and normalizes images for model input.</li>
        <li><strong>Prediction</strong>: Classifies the input image into one of 30 constellations.</li>
        <li><strong>Graph Image Loading</strong>: Fetches a corresponding graph representation of the detected constellation.</li>
        <li><strong>Class Information</strong>: Retrieves detailed information about the detected constellation from a JSON file.</li>
        <li><strong>GUI</strong>: Implements a user-friendly interface using Tkinter to display the original image, graph, and textual details.</li>
    </ul>

    <h4>How to Run:</h4>
    <ol>
        <li>Ensure the required dependencies (<code>cv2</code>, <code>tensorflow</code>, <code>numpy</code>, <code>tkinter</code>, <code>PIL</code>) are installed.</li>
        <li>Place the constellation graph images in the <code>Normalised_Templates</code> folder and the constellation information in <code>constellation_info.json</code>.</li>
        <li>Run the script:
            <pre><code>python one.py</code></pre>
        </li>
    </ol>

    <h4>Key Files:</h4>
    <ul>
        <li><strong>Saved Model Directory</strong>: <code>saved_model/</code></li>
        <li><strong>Graph Folder</strong>: <code>Normalised_Templates/</code></li>
        <li><strong>Constellation Info JSON</strong>: <code>constellation_info.json</code></li>
    </ul>

    <h3><code>preprocess.py</code>: Image Preprocessing Module</h3>
    <p>This script prepares input images for graph-based constellation detection.</p>

    <h4>Key Functionalities:</h4>
    <ul>
        <li><strong>Feature Extraction</strong>: Extracts keypoints using ORB (Oriented FAST and Rotated BRIEF).</li>
        <li><strong>Graph Representation</strong>: Converts extracted features into graph structures.</li>
        <li><strong>Normalization</strong>: Scales features for consistent input to the detection model.</li>
    </ul>

    <h4>Usage:</h4>
    <pre><code>from preprocess import ConstellationDetection

constellation_detector = ConstellationDetection("g")
constellation_detector.process_image(image_path)
</code></pre>

    <h4>Dependencies:</h4>
    <ul>
        <li><strong>cv2</strong>: For image processing and feature extraction.</li>
        <li><strong>numpy</strong>: For numerical computations.</li>
    </ul>

    <h3><code>gnn.py</code>: Graph Neural Network Model</h3>
    <p>This script trains and applies a GNN for constellation detection. It uses weighted nodes and multi-head attention mechanisms for better performance.</p>

    <h4>Key Functionalities:</h4>
    <ul>
        <li><strong>Graph Construction</strong>: Builds graphs from star coordinates and connections.</li>
        <li><strong>GNN Training</strong>: Trains a GNN model to classify constellations based on graph features.</li>
        <li><strong>Prediction</strong>: Uses the trained model to predict the constellation class from input graphs.</li>
    </ul>

    <h4>Usage:</h4>
    <ol>
        <li>Ensure the required libraries (<code>torch</code>, <code>torch_geometric</code>) are installed.</li>
        <li>Train or load the GNN model and use it as follows:
            <pre><code>from gnn import ConstellationGNN

model = ConstellationGNN()
# Training and prediction processes</code></pre>
        </li>
    </ol>

    <h4>Dependencies:</h4>
    <ul>
        <li><strong>torch</strong>: For deep learning functionalities.</li>
        <li><strong>torch_geometric</strong>: For graph-based learning and operations.</li>
    </ul>

    <h2>Installation and Setup</h2>
    <ol>
        <li>Clone the repository and navigate to the project folder:
            <pre><code>git clone &lt;repository-url&gt;
cd &lt;repository-name&gt;</code></pre>
        </li>
        <li>Install dependencies:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Place all necessary assets (model, graph images, JSON) in their respective directories.</li>
    </ol>

    <h2>Running the Application</h2>
    <p>Start the GUI application by executing <code>one.py</code>:</p>
    <pre><code>python one.py</code></pre>

    <h2>Future Work</h2>
    <ul>
        <li>Enhance the GNN model for improved accuracy.</li>
        <li>Add support for additional constellations.</li>
        <li>Enable real-time image capture and detection.</li>
    </ul>

    <h2>License</h2>
    <p>This project is open-source and available under the MIT License.</p>

</body>
</html>
