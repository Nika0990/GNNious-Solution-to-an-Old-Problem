# GNNious Solution to an Old Problem
## Authors
- Diana Morgan 
- Sergiy Horef 
- Veronika Sorochenkova 

## Project Overview
Finding the right job is a complex challenge that often relies on inefficient manual screening. This project introduces an innovative approach using Graph Neural Networks (GNNs) to improve job matching efficiency. Our solution:

- Represents each user profile as a graph with two types of nodes: the user and their previous job positions
- Uses GNNs to generate feature vectors that align user profiles with suitable job positions
- Employs cosine similarity for matching profiles with potential job opportunities
- Enhances job recommendations by capturing deeper relationships between users and jobs
- Improves efficiency and reduces job search time

### Key Features
- Graph-based profile representation
- Advanced GNN-based feature learning
- Intelligent job matching using cosine similarity
- Temporal awareness of career progression

## Technical Methodology

### Data Collection and Integration
Our dataset combines multiple sources:
- Original user profile data from Bright Data
- Web-scraped job postings from Indeed.com using Bright Data's Scraping Browser proxy
- Additional open-source job datasets (see [External Datasets](#external-datasets))

Total dataset size: 133,834 job postings (3,780 scraped + external sources)

### Feature Selection
1. **Job Posting Features**:
   - Company name
   - Position title
   - Job location
   - Job description

2. **Profile Features**:
   - About section
   - City
   - Education
   - Experience

### AI Implementation
1. **Initial Embeddings**:
   - Single column text representation via concatenation
   - BERT model (spark-nlp) embedding generation
   - 1024-dimensional vectors (512 word limit)

2. **Graph Neural Network Architecture**:
   - **Node Types**:
     - Profile node (user information)
     - Job position nodes (previous experiences)
     - Phantom nodes (temporal information handlers)
   - **Edge Structure**: Directed edges from job positions to profile through phantoms
   - **Model**: Graph Isomorphism Network (GIN) for maximum expressiveness

3. **Training Process**:
   - Sample size: ~100 informative profiles (computational constraints)
   - 80-20 train-test split
   - Novel loss function incorporating temporal job relevance
   - Negative sampling for balanced training

## Repository Structure
```
├── companies.csv                    # List of companies used for data scraping
├── data_exploration.ipynb          # Jupyter notebook for initial data exploration
├── data_scraping.py               # Python script for scraping job listings data
├── project_proposal.tex           # LaTeX source for the project proposal
├── requirements.txt               # Python dependencies
├── Databricks Notebook/          # Contains the main analysis notebooks
│   ├── Final_Project.dbc         # Databricks archive format
│   └── Final_Project.ipynb       # IPython notebook format
└── Pictures/                     # Visualization outputs
    ├── job_postings_example.png
    ├── loss.png
    ├── profiles_columns.png
    └── various analysis plots
```

## Setup and Installation

### Prerequisites
1. Python 3.x
2. Required Python packages (install using `pip install -r requirements.txt`)
3. Authentication key for the API (stored in `auth_key.txt`)
4. List of target companies (in `companies.txt`)

### Data Collection
1. Place your API authentication key in `auth_key.txt`
2. Ensure your target companies list is in `companies.txt`
3. Run the scraping script:
   ```bash
   python data_scraping.py
   ```

### Data Exploration
1. After data collection, you can explore the gathered data using:
   ```bash
   jupyter notebook data_exploration.ipynb
   ```

## Data Processing Pipeline
1. **Data Collection**: Using custom scraping scripts to gather job listings data
2. **Data Exploration**: Initial analysis and cleaning in Jupyter notebooks
3. **Advanced Analysis**: Main processing and modeling in Databricks environment
4. **Visualization**: Results visualization and interpretation (available in the Pictures directory)

## Results and Evaluation

### Key Findings
- Successfully predicted relevant job positions for both experienced and entry-level candidates
- Model demonstrated ability to understand career progression
- Example predictions:
  - For experienced candidate: "Change Management Coordinator"
  - For entry-level candidate with coding experience: "Product Manager - Data Science"

### Evaluation Method
- Custom loss function measuring prediction accuracy against temporal job relevance
- Baseline comparison with null model (zero vector output)
- Test set validation showing significant improvement over baseline

### Limitations
1. **Computational Resources**:
   - Limited dataset size due to cluster constraints
   - Simplified graph structure (single-profile vs. company-connected graphs)
   - Restricted training sample size

2. **Technical Constraints**:
   - Hardware limitations affecting model complexity
   - Focus on Databricks/Spark implementation for scalability

## External Datasets
The project incorporates the following public datasets:
- [Linkedin 124k](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) - 124k job postings
- [Tech Linkedin 82k](https://github.com/example/tech-linkedin) - 82k tech job postings
- [Glassdoor 4k](https://github.com/example/glassdoor) - 4k job postings

## Project Documentation
- Final project report: See `GNN_Project_report.pdf`
- Visual results and plots are stored in the `Pictures/` directory

## Poster
[![Poster Preview](Pictures/poster.png)](poster.pdf)  

## Contact
For any questions or clarifications about this project, please contact any of the authors listed above.
