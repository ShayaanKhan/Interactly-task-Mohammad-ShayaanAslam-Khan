## Installation

1. **Set up a Python virtual environment**:
   python3 -m venv env
   source env/bin/activate

2. **Install the required packages**:
   pip install -r requirements.txt


## Usage

### Step 1: Prepare Data

Ensure your candidate data and training data are available in the `data/` directory

### Step 2: Train the Language Model

Fine-tune the language model on your dataset:

python job_matcher/scripts/train_llm.py

### Step 3: Run the Command-Line Interface
Use the CLI to input a job description and retrieve matching candidate profiles:

python job_matcher/scripts/cli.py

File Descriptions
models/:
   fine_tuned_model/: Directory containing the fine-tuned language model.

scripts/:
   cli.py: Command-line interface script to interact with the RAG system.
   preprocess_data.py: Script for preprocessing candidate data.
   index_data.py: Script for indexing candidate data using FAISS.
   rag_framework.py: Script implementing the RAG framework.
   train_llm.py: Script for fine-tuning the language model.
   requirements.txt: List of required Python packages.

Dependencies:
   transformers
   sentence-transformers
   faiss-cpu
   pandas
   numpy
   scikit-learn

Install these dependencies using the requirements.txt file provided.
