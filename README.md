# Job Description Matching System

This project is a job description matching system that uses a fine-tuned BERT model, Elasticsearch for candidate profile retrieval, and a Flask backend with a React frontend.

## Setup and Installation

### Prerequisites

- Python 3.7+
- Node.js and npm
- Elasticsearch
- MongoDB

### Backend Setup

1. **Clone the repository**

   
   git clone https://github.com/yourusername/job-matching-system.git
   cd job-matching-system/backend
   

2. **Install Python dependencies**

   
   pip install -r requirements.txt
   

3. **Set Up Elasticsearch and MongoDB**

   Make sure you have Elasticsearch and MongoDB running locally.

4. **Database Setup**

   Run the `database_setup.py` script to load and index the candidate data.

   
   python database_setup.py
   

5. **Model Fine-Tuning**

   Run the `model_finetuning.py` script to fine-tune the BERT model using the provided `resumes.csv` dataset.

   
   python model_finetuning.py
   

6. **Start the Flask Application**

   Run the Flask application.

   
   python app.py
   

### Frontend Setup

1. **Navigate to the frontend directory**

   
   cd ../frontend/src
   

2. **Install Node.js dependencies**

   
   npm install
   

3. **Start the React application**

   
   npm start
   

## Usage

1. **Open the frontend application**

   Go to `http://localhost:3000` in your web browser.

2. **Enter a Job Description**

   Enter a job description in the input field and click "Search".

3. **View Matching Candidates**

   The application will display a list of matching candidates based on the job description.

## File Descriptions

- **backend/app.py**: The main Flask application file that handles API requests.
- **backend/database_setup.py**: Script to load and index candidate data into MongoDB and Elasticsearch.
- **backend/model_finetuning.py**: Script to fine-tune the BERT model using the `resumes.csv` dataset.
- **backend/retrieval.py**: Contains functions for retrieving candidate profiles and generating responses.
- **backend/requirements.txt**: Lists the Python dependencies required for the backend.
- **frontend/src/App.js**: The main React component for the frontend application.
- **frontend/src/index.js**: The entry point for the React application.
- **frontend/package.json**: Lists the Node.js dependencies required for the frontend.
- **data/RecruterPilot candidate sample input dataset - Sheet1.csv**: Sample input dataset containing candidate profiles.
- **data/resumes.csv**: Dataset used for fine-tuning the BERT model.
