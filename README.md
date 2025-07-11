# Indian Name Correction using Jaro-Winkler Similarity

## Getting Started
Follow these steps to set up and run the project:

### 1. Install Dependencies
```bash
# Create and activate a virtual environment
python3 -m venv .venv
# On Unix/MacOS:
source .venv/bin/activate
# On Windows:
. .\.venv\Scripts\Activate.ps1
# Install required packages
pip install -r requirements.txt
```
---

## Problem Statement
This project deals with an **unlabelled dataset of Indian names**, where the goal is to identify and correct potentially miswritten names (e.g., due to typos or transliteration issues).

---

## Approach

### 1. **Data Collection**:
   - Used a publicly available dataset of Indian first names from Kaggle:  
     👉 [Indian Names Dataset by Anany Sharma](https://www.kaggle.com/datasets/ananysharma/indian-names-dataset/data)
     👉 [Indian Names Corpus](https://www.kaggle.com/datasets/jasleensondhi/indian-names-corpus-nltk-data)

### 2. **Preprocessing**:
   - Cleaned and standardized the sample name dataset for accurate matching.
   - Removed pronouns and titles (Mr., Mrs., Ms., Dr., etc.) from input names.
   - Normalized whitespace and capitalization for consistent matching.
   - Removed indexing or other metadata to focus solely on raw name data.

### 3. **Fuzzy Matching with Jaro-Winkler Similarity**:
   - Applied **Jaro-Winkler Similarity** algorithm to determine similarity between mislabelled names and correctly spelled ones.
   - **Why Jaro-Winkler over Levenshtein?**
     - Specifically designed for name matching and performs better on proper names
     - Gives higher similarity scores to strings that match from the beginning (common prefix bias)
     - Better handles transposition errors (character swaps) which are common in name typos
     - More effective for Indian names with transliteration variations (e.g., Rajiv vs Rajeev)
   - Similarity scores range from 0.0 to 1.0, where 1.0 indicates a perfect match

### 4. **Mapping Strategy**:
   - For each unlabelled or mislabelled name, compute Jaro-Winkler similarity scores against the reference dataset.
   - Select the closest matching name with the **highest similarity score** above a defined threshold (default: 0.8).
   - If no match meets the threshold, retain the original name to avoid false corrections.

### 5. **Interactive Input Options**:
   - **Manual Input Mode**: Users can enter individual names for correction
   - **Batch Processing Mode**: Process existing CSV files with multiple names
   - Automatic CSV generation for different input methods

### 6. **Output**:
   - A corrected mapping of names that aligns noisy data with likely intended values
   - Detailed similarity scores for transparency and quality assessment
   - Comprehensive correction statistics and examples
   - Helps in downstream tasks like deduplication, user profiling, or dataset augmentation

---

## Usage

### Running the Script
```bash
python main-jaro-winkler.py
```

### Interactive Options
1. **Manual Input**: Enter individual names for correction
2. **Batch Processing**: Process existing `test.csv` file

### Output Files
- `prompted_input.csv`: Generated when using manual input
- `prompted_results_jaro_winkler.csv`: Results for manual input
- `test_results_jaro_winkler.csv`: Results for batch processing

### Sample Output Structure
```csv
Corrected First Name,Corrected Last Name,First Name,Last Name,Gender,*Email,*Date of Birth,First Name Similarity,Last Name Similarity
Rajiv,Sharma,Mr. Rajivv,Sharmaa,Male,*rajiv.sharma@example.com,*8/15/1985,0.952,0.909
Priyanka,Patel,Mrs. Pryanka,Patell,Female,*pryanka.patel@example.com,*12/25/1990,0.889,0.800
```

---

## Algorithm Comparison

| Feature | Levenshtein Distance | Jaro-Winkler Similarity |
|---------|---------------------|-------------------------|
| **Focus** | Edit distance | Name similarity |
| **Scoring** | Lower is better (0+) | Higher is better (0.0-1.0) |
| **Prefix Bias** | No | Yes (better for names) |
| **Transpositions** | Penalizes heavily | Handles better |
| **Indian Names** | Basic | Superior for transliteration |
| **Interpretability** | Less intuitive | More intuitive |

---

## Configuration

### Similarity Threshold
- **0.9+**: Very strict matching (only minor typos)
- **0.8**: Good balance (default) - handles most typos and variations
- **0.7**: More lenient - catches more variations but may have false positives
- **0.6**: Very lenient - use with caution

### File Structure
```
project/
├── main-jaro-winkler.py          # Main script
├── first_name.txt             # Reference first names
├── last_name.txt              # Reference last names
├── test.csv                   # Input data for batch processing
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Notes

- This is an **unsupervised name correction task**.
- **Jaro-Winkler similarity** provides a more effective way to handle typographical and phonetic errors in names compared to traditional edit distance methods.
- **Specifically optimized for Indian names** with common transliteration variations.
- **Interactive and batch processing modes** for different use cases.
- **Comprehensive logging and statistics** for quality assessment.
- Further enhancements can include phonetic matching (Soundex, Metaphone) or using ML models trained on name datasets.

---

## Example Corrections

| Original | Corrected | Similarity Score |
|----------|-----------|------------------|
| Rajivv | Rajiv | 0.952 |
| Pryanka | Priyanka | 0.889 |
| Anjli | Anjali | 0.900 |
| Sharmaa | Sharma | 0.909 |

---

## Future Enhancements

1. **Hybrid Approach**: Combine Jaro-Winkler with phonetic algorithms
2. **Machine Learning**: Train models on Indian name datasets
3. **Regional Variations**: Handle different Indian language transliterations
4. **Confidence Scoring**: Multi-tier confidence levels for corrections
5. **API Integration**: RESTful API for real-time name correction