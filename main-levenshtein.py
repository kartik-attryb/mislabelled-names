import pandas as pd
import re
from Levenshtein import distance
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NameCorrector:
    """
    A class to handle name correction using Levenshtein distance matching.
    """
    
    def __init__(self, first_names_file, last_names_file):
        """
        Initialize the NameCorrector with reference name datasets.
        
        Args:
            first_names_file (str): Path to the first names reference file
            last_names_file (str): Path to the last names reference file
        """
        self.first_names = self._load_reference_names(first_names_file)
        self.last_names = self._load_reference_names(last_names_file)
        logger.info(f"Loaded {len(self.first_names)} first names and {len(self.last_names)} last names")
    
    def _load_reference_names(self, file_path):
        """
        Load reference names from a text file.
        
        Args:
            file_path (str): Path to the reference names file
            
        Returns:
            list: List of cleaned reference names
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                names = [line.strip() for line in file if line.strip()]
            return [self._clean_name(name) for name in names]
        except FileNotFoundError:
            logger.error(f"Reference file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading reference file {file_path}: {str(e)}")
            raise
    
    def _clean_name(self, name):
        """
        Clean a name by removing pronouns, extra whitespace, and normalizing case.
        
        Args:
            name (str): Raw name to clean
            
        Returns:
            str: Cleaned name
        """
        if not isinstance(name, str):
            return ""
        
        # Remove common pronouns/titles
        pronouns = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'miss', 'master']
        cleaned = name.lower().strip()
        
        # Remove pronouns
        for pronoun in pronouns:
            if cleaned.startswith(pronoun):
                cleaned = cleaned[len(pronoun):].strip()
                break
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Capitalize first letter of each word
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())
        
        return cleaned
    
    def _find_best_match(self, target_name, reference_names, threshold=None):
        """
        Find the best matching name using Levenshtein distance.
        
        Args:
            target_name (str): Name to match
            reference_names (list): List of reference names
            threshold (int, optional): Maximum edit distance threshold
            
        Returns:
            tuple: (best_match, distance) or (None, None) if no good match found
        """
        if not target_name or not reference_names:
            return None, None
        
        best_match = None
        min_distance = float('inf')
        
        for ref_name in reference_names:
            dist = distance(target_name.lower(), ref_name.lower())
            if dist < min_distance:
                min_distance = dist
                best_match = ref_name
        
        # Apply threshold if specified
        if threshold is not None and min_distance > threshold:
            return None, None
        
        return best_match, min_distance
    
    def correct_names(self, df, first_name_col='First Name', last_name_col='Last Name', 
                     threshold=None):
        """
        Correct names in a DataFrame using reference datasets.
        
        Args:
            df (pd.DataFrame): Input DataFrame with names to correct
            first_name_col (str): Column name for first names
            last_name_col (str): Column name for last names
            threshold (int, optional): Maximum edit distance threshold
            
        Returns:
            pd.DataFrame: DataFrame with corrected names
        """
        df_corrected = df.copy()
        
        # Initialize new columns for corrected names
        df_corrected['Corrected First Name'] = ''
        df_corrected['Corrected Last Name'] = ''
        df_corrected['First Name Distance'] = 0
        df_corrected['Last Name Distance'] = 0
        
        logger.info(f"Processing {len(df)} names for correction")
        
        for idx, row in df.iterrows():
            # Clean original names
            original_first = self._clean_name(str(row[first_name_col]))
            original_last = self._clean_name(str(row[last_name_col]))
            
            # Find best matches
            corrected_first, first_dist = self._find_best_match(
                original_first, self.first_names, threshold
            )
            corrected_last, last_dist = self._find_best_match(
                original_last, self.last_names, threshold
            )
            
            # Store results
            df_corrected.loc[idx, 'Corrected First Name'] = corrected_first or original_first
            df_corrected.loc[idx, 'Corrected Last Name'] = corrected_last or original_last
            df_corrected.loc[idx, 'First Name Distance'] = first_dist if first_dist is not None else 0
            df_corrected.loc[idx, 'Last Name Distance'] = last_dist if last_dist is not None else 0
            
            if first_dist is not None and first_dist > 0:
                logger.info(f"First name correction: '{original_first}' -> '{corrected_first}' (distance: {first_dist})")
            if last_dist is not None and last_dist > 0:
                logger.info(f"Last name correction: '{original_last}' -> '{corrected_last}' (distance: {last_dist})")
        
        return df_corrected
    
    def generate_final_csv(self, input_csv, output_csv='final.csv', threshold=None):
        """
        Process the input CSV and generate the final corrected CSV.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str): Path to output CSV file
            threshold (int, optional): Maximum edit distance threshold
        """
        try:
            # Load input data
            df = pd.read_csv(input_csv)
            logger.info(f"Loaded input data with {len(df)} records")
            
            # Correct names
            df_corrected = self.correct_names(df, threshold=threshold)
            
            # Reorder columns to have corrected names first
            original_cols = df.columns.tolist()
            new_cols = ['Corrected First Name', 'Corrected Last Name'] + original_cols + ['First Name Distance', 'Last Name Distance']
            df_final = df_corrected[new_cols]
            
            # Save to CSV
            df_final.to_csv(output_csv, index=False)
            logger.info(f"Final corrected data saved to {output_csv}")
            
            # Print summary statistics
            self._print_correction_summary(df_corrected)
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise
    
    def _print_correction_summary(self, df):
        """
        Print summary statistics about the corrections made.
        
        Args:
            df (pd.DataFrame): DataFrame with correction results
        """
        first_name_corrections = (df['First Name Distance'] > 0).sum()
        last_name_corrections = (df['Last Name Distance'] > 0).sum()
        total_records = len(df)
        
        print("\n" + "="*50)
        print("CORRECTION SUMMARY")
        print("="*50)
        print(f"Total records processed: {total_records}")
        print(f"First name corrections made: {first_name_corrections}")
        print(f"Last name corrections made: {last_name_corrections}")
        print(f"First name correction rate: {(first_name_corrections/total_records)*100:.1f}%")
        print(f"Last name correction rate: {(last_name_corrections/total_records)*100:.1f}%")
        print("="*50)

def main():
    """
    Main function to run the name correction process.
    """
    # File paths
    FIRST_NAMES_FILE = 'first_name.txt'
    LAST_NAMES_FILE = 'last_name.txt'
    INPUT_CSV = 'test.csv'
    OUTPUT_CSV = 'test_results_levenshtein.csv'
    
    # Optional: Set a threshold for maximum edit distance (None for no threshold)
    DISTANCE_THRESHOLD = None  # You can set this to a number like 3 if needed
    
    try:
        # Initialize the name corrector
        corrector = NameCorrector(FIRST_NAMES_FILE, LAST_NAMES_FILE)
        
        # Process the CSV and generate final output
        corrector.generate_final_csv(INPUT_CSV, OUTPUT_CSV, threshold=DISTANCE_THRESHOLD)
        
        print(f"\nName correction completed successfully!")
        print(f"Check the output file: {OUTPUT_CSV}")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()