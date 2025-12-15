🛍 SmartShopper AI: Machine Learning-Powered Deal Finder

Project Overview

SmartShopper AI is a sophisticated e-commerce analysis tool that integrates cutting-edge Visual Search (Deep Learning) with Sentiment Analysis (NLP) to provide users with the best market deals and genuine product quality insights.

The system's primary goal is to solve the problem of visual product discovery and provide verifiable quality metrics, moving beyond simple text search and star ratings.

Key Features

Visual Similarity Search: Upload any product image (e.g., a photo of a skirt), and the system finds the most visually similar items in the catalog using a powerful CLIP-based Vector Index.

Zero-Shot Classification: Accurately classifies and labels items (e.g., distinguishing between a "Skirt" and a "Bra") by leveraging a large, diverse product catalog sourced from the Kaggle Fashion Product Images Dataset.

Hybrid Data Aggregation: Aggregates and compares dynamic mock price data, discounts, and ratings from multiple platforms (simulating Amazon, Flipkart, Meesho) for competitive analysis.

NLP Sentiment Analysis: Analyzes generated customer reviews using TextBlob to calculate a reliable sentiment score, providing a critical quality metric on product feedback.

Modular Architecture: Built on a scalable, modular architecture in Python/Streamlit, separating the UI, ML logic, data simulation, and NLP into distinct services.

⚙️ Technology Stack

Component

Technology

Role

Frontend/UI

Streamlit

Interactive web application interface.

Visual Search

CLIP Model

Generates high-dimensional vector embeddings for images.

Vector Index

NumPy (.npy) & Cosine Similarity

Stores embeddings for high-speed similarity search.

Sentiment Analysis

TextBlob

Calculates review polarity and provides a verbal verdict.

Data Source

Kaggle Product Images (styles.csv)

Provides a large, real-world image catalog for high ML accuracy.

🚀 Setup and Installation

Follow these steps to run the project locally.

Prerequisites

Python 3.8+

All project files from the GitHub repository.

Step 1: Clone the Repository

git clone [https://github.com/chhayachaudhary/SmartShopper-AI-Visual-Search.git](https://github.com/chhayachaudhary/SmartShopper-AI-Visual-Search.git)
cd SmartShopper-AI-Visual-Search


Step 2: Virtual Environment Setup & Dependencies

# Assuming the environment is already set up and dependencies are installed
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


Step 3: Acquire and Place Data (CRITICAL)

Download Metadata: Place the styles.csv metadata file in the root project directory.

Download Images: Place a sample of images (e.g., 200 items) inside the data/product_images/ folder.

Step 4: Run the Application

streamlit run app.py


Index the Catalog: Click "Re-Index Product Catalog" to build the vector index (limited to 200 items for speed).

Test: Upload an image to test the visual search and price analysis features.
