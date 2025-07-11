import pandas as pd
import re
from jellyfish import jaro_winkler_similarity
import logging
import os
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NameCorrector:
    """
    A class to handle name correction using Jaro-Winkler similarity matching.
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
    
    def _find_best_match(self, target_name, reference_names, threshold=0.8):
        """
        Find the best matching name using Jaro-Winkler similarity.
        
        Args:
            target_name (str): Name to match
            reference_names (list): List of reference names
            threshold (float): Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            tuple: (best_match, similarity_score) or (None, 0.0) if no good match found
        """
        if not target_name or not reference_names:
            return None, 0.0
        
        best_match = None
        max_similarity = 0.0
        
        for ref_name in reference_names:
            # Calculate Jaro-Winkler similarity (returns value between 0 and 1)
            similarity = jaro_winkler_similarity(target_name.lower(), ref_name.lower())
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = ref_name
        
        # Apply threshold - only return match if similarity is above threshold
        if max_similarity >= threshold:
            return best_match, max_similarity
        else:
            return None, max_similarity
    
    def correct_names(self, df, first_name_col='First Name', last_name_col='Last Name', 
                     threshold=0.8):
        """
        Correct names in a DataFrame using reference datasets.
        
        Args:
            df (pd.DataFrame): Input DataFrame with names to correct
            first_name_col (str): Column name for first names
            last_name_col (str): Column name for last names
            threshold (float): Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            pd.DataFrame: DataFrame with corrected names
        """
        df_corrected = df.copy()
        
        # Initialize new columns for corrected names
        df_corrected['Corrected First Name'] = ''
        df_corrected['Corrected Last Name'] = ''
        df_corrected['First Name Similarity'] = 0.0
        df_corrected['Last Name Similarity'] = 0.0
        
        logger.info(f"Processing {len(df)} names for correction with threshold {threshold}")
        
        for idx, row in df.iterrows():
            # Clean original names
            original_first = self._clean_name(str(row[first_name_col]))
            original_last = self._clean_name(str(row[last_name_col]))
            
            # Find best matches
            corrected_first, first_similarity = self._find_best_match(
                original_first, self.first_names, threshold
            )
            corrected_last, last_similarity = self._find_best_match(
                original_last, self.last_names, threshold
            )
            
            # Store results - use original if no good match found
            df_corrected.loc[idx, 'Corrected First Name'] = corrected_first or original_first
            df_corrected.loc[idx, 'Corrected Last Name'] = corrected_last or original_last
            df_corrected.loc[idx, 'First Name Similarity'] = first_similarity
            df_corrected.loc[idx, 'Last Name Similarity'] = last_similarity
            
            # Log corrections made
            if corrected_first and corrected_first != original_first:
                logger.info(f"First name correction: '{original_first}' -> '{corrected_first}' (similarity: {first_similarity:.3f})")
            if corrected_last and corrected_last != original_last:
                logger.info(f"Last name correction: '{original_last}' -> '{corrected_last}' (similarity: {last_similarity:.3f})")
        
        return df_corrected
    
    def generate_final_csv(self, input_csv, output_csv='final.csv', threshold=0.8):
        """
        Process the input CSV and generate the final corrected CSV.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str): Path to output CSV file
            threshold (float): Minimum similarity threshold (0.0 to 1.0)
        """
        try:
            # Load input data
            df = pd.read_csv(input_csv)
            logger.info(f"Loaded input data with {len(df)} records")
            
            # Correct names
            df_corrected = self.correct_names(df, threshold=threshold)
            
            # Reorder columns to have corrected names first
            original_cols = df.columns.tolist()
            new_cols = ['Corrected First Name', 'Corrected Last Name'] + original_cols + ['First Name Similarity', 'Last Name Similarity']
            df_final = df_corrected[new_cols]
            
            # Save to CSV
            df_final.to_csv(output_csv, index=False)
            logger.info(f"Final corrected data saved to {output_csv}")
            
            # Print summary statistics
            self._print_correction_summary(df_corrected, threshold)
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise
    
    def _print_correction_summary(self, df, threshold):
        """
        Print summary statistics about the corrections made.
        
        Args:
            df (pd.DataFrame): DataFrame with correction results
            threshold (float): Similarity threshold used
        """
        # Count corrections made (where similarity >= threshold and name was changed)
        first_name_corrections = 0
        last_name_corrections = 0
        
        for idx, row in df.iterrows():
            original_first = self._clean_name(str(row['First Name']))
            original_last = self._clean_name(str(row['Last Name']))
            
            if (row['First Name Similarity'] >= threshold and 
                row['Corrected First Name'] != original_first):
                first_name_corrections += 1
                
            if (row['Last Name Similarity'] >= threshold and 
                row['Corrected Last Name'] != original_last):
                last_name_corrections += 1
        
        total_records = len(df)
        avg_first_similarity = df['First Name Similarity'].mean()
        avg_last_similarity = df['Last Name Similarity'].mean()
        
        print("\n" + "="*60)
        print("JARO-WINKLER CORRECTION SUMMARY")
        print("="*60)
        print(f"Total records processed: {total_records}")
        print(f"Similarity threshold used: {threshold}")
        print(f"First name corrections made: {first_name_corrections}")
        print(f"Last name corrections made: {last_name_corrections}")
        print(f"First name correction rate: {(first_name_corrections/total_records)*100:.1f}%")
        print(f"Last name correction rate: {(last_name_corrections/total_records)*100:.1f}%")
        print(f"Average first name similarity: {avg_first_similarity:.3f}")
        print(f"Average last name similarity: {avg_last_similarity:.3f}")
        print("="*60)
        
        # Show some examples of corrections
        corrections_made = []
        for idx, row in df.iterrows():
            original_first = self._clean_name(str(row['First Name']))
            original_last = self._clean_name(str(row['Last Name']))
            
            if (row['First Name Similarity'] >= threshold and 
                row['Corrected First Name'] != original_first):
                corrections_made.append(f"First: '{original_first}' -> '{row['Corrected First Name']}' ({row['First Name Similarity']:.3f})")
            
            if (row['Last Name Similarity'] >= threshold and 
                row['Corrected Last Name'] != original_last):
                corrections_made.append(f"Last: '{original_last}' -> '{row['Corrected Last Name']}' ({row['Last Name Similarity']:.3f})")
        
        if corrections_made:
            print("\nSample corrections made:")
            for correction in corrections_made[:5]:  # Show first 5 corrections
                print(f"  - {correction}")
            if len(corrections_made) > 5:
                print(f"  ... and {len(corrections_made) - 5} more corrections")

def display_corrected_names(output_csv):
    """
    Display the corrected names in a formatted console output.
    
    Args:
        output_csv (str): Path to the output CSV file with corrected names
    """
    try:
        # Read the results CSV
        df = pd.read_csv(output_csv)
        
        print("\n" + "="*80)
        print("CORRECTED NAMES - CONSOLE OUTPUT")
        print("="*80)
        
        for idx, row in df.iterrows():
            original_first = row['First Name']
            original_last = row['Last Name']
            corrected_first = row['Corrected First Name']
            corrected_last = row['Corrected Last Name']
            first_similarity = row['First Name Similarity']
            last_similarity = row['Last Name Similarity']
            
            print(f"\nRecord {idx + 1}:")
            print(f"  Original Names : {original_first} {original_last}")
            print(f"  Corrected Names: {corrected_first} {corrected_last}")
            print(f"  Similarity Scores: First={first_similarity:.3f}, Last={last_similarity:.3f}")
            
            # Highlight if corrections were made
            if corrected_first != original_first.replace('Mr. ', '').replace('Mrs. ', '').replace('Ms. ', '').replace('Dr. ', '').replace('Shri. ', '').replace('Smt. ', '').strip():
                print(f"  ✓ First name corrected: '{original_first}' → '{corrected_first}'")
            if corrected_last != original_last.strip():
                print(f"  ✓ Last name corrected: '{original_last}' → '{corrected_last}'")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error displaying corrected names: {str(e)}")
        print(f"Error reading results file: {str(e)}")

def create_prompted_input_csv(first_name, last_name, output_file='prompted_input.csv'):
    """
    Create a CSV file with manually entered names.
    
    Args:
        first_name (str): First name entered by user
        last_name (str): Last name entered by user
        output_file (str): Output CSV file path
    """
    # Create a simple DataFrame with the entered names
    # Using similar structure to test.csv but with minimal required columns
    data = {
        'First Name': [first_name],
        'Last Name': [last_name],
        'Gender': ['Unknown'],  # Default value
        '*Email': ['*manual.input@example.com'],  # Default value
        '*Date of Birth': ['*01/01/2000']  # Default value
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Created prompted input CSV: {output_file}")
    print(f"Created input file: {output_file}")
    return output_file

def get_user_input():
    """
    Get user input for manual name entry.

    Returns:
        tuple: (use_manual_input, first_name, last_name)
    """
    print("\n" + "="*60)
    print("NAME CORRECTION TOOL - INPUT SELECTION")
    print("="*60)

    while True:
        choice = input("\nDo you want to enter names manually? (Y/N) (test will run on test.csv if chosen N): ").strip().upper()

        if choice in ['Y', 'YES']:
            print("\nEntering manual input mode...")

            # Get first name
            while True:
                first_name = input("Enter the first name to correct: ").strip()
                if first_name:
                    break
                print("Please enter a valid first name.")

            # Get last name
            while True:
                last_name = input("Enter the last name to correct: ").strip()
                if last_name:
                    break
                print("Please enter a valid last name.")

            print(f"\nYou entered:")
            print(f"First Name: {first_name}")
            print(f"Last Name: {last_name}")

            return True, first_name, last_name

        elif choice in ['N', 'NO']:
            print("\nUsing existing test.csv file for processing...")
            print("WARNING: Processing will be done on test.csv")
            return False, None, None

        else:
            print("Please enter 'Y' for Yes or 'N' for No.")

def append_name_to_file(filename, name):
    """
    Append a name to a text file, ensuring it's written on a new line.
    """
    with open(filename, 'a', encoding='utf-8') as f:
        if os.path.getsize(filename) > 0:
            f.write('\n')  # Add newline if file already has content
        f.write(name.strip())

def main():
    """
    Main function to run the name correction process.
    """
    # File paths
    FIRST_NAMES_FILE = 'first_name.txt'
    LAST_NAMES_FILE = 'last_name.txt'
    DEFAULT_INPUT_CSV = 'test.csv'
    PROMPTED_INPUT_CSV = 'prompted_input.csv'

    # Similarity threshold (0.0 to 1.0)
    SIMILARITY_THRESHOLD = 0.8

    try:
        # Get user input preference
        use_manual_input, first_name, last_name = get_user_input()

        # Determine input CSV file
        if use_manual_input:
            input_csv = create_prompted_input_csv(first_name, last_name, PROMPTED_INPUT_CSV)
            output_csv = 'prompted_results_jaro_winkler.csv'
        else:
            input_csv = DEFAULT_INPUT_CSV
            output_csv = 'test_results_jaro_winkler.csv'

        # Initialize the name corrector
        corrector = NameCorrector(FIRST_NAMES_FILE, LAST_NAMES_FILE)

        # Process the CSV and generate final output
        corrector.generate_final_csv(input_csv, output_csv, threshold=SIMILARITY_THRESHOLD)

        # Display corrected names in console
        display_corrected_names(output_csv)

        print(f"\nName correction completed successfully using Jaro-Winkler similarity!")
        print(f"Input file used: {input_csv}")
        print(f"Check the output file: {output_csv}")
        print(f"\nNote: Similarity scores range from 0.0 to 1.0")
        print(f"      1.0 = Perfect match, 0.0 = No similarity")
        print(f"      Threshold used: {SIMILARITY_THRESHOLD}")

        # Ask for feedback if manual input was used
        if use_manual_input:
            feedback = input("\nWas the corrected result accurate? (Y/N): ").strip().upper()
            if feedback in ['N', 'NO']:
                print("\nPlease provide the correct values.")
                corrected_first = input("Correct First Name: ").strip().capitalize()
                corrected_last = input("Correct Last Name: ").strip().capitalize()

                # Save the corrected names into respective files
                append_name_to_file(FIRST_NAMES_FILE, corrected_first)
                append_name_to_file(LAST_NAMES_FILE, corrected_last)

                print(f"\n✅ Corrected names saved to '{FIRST_NAMES_FILE}' and '{LAST_NAMES_FILE}'.")
                
            elif feedback in ['Y', 'YES']:
                print("\n✅ Great! The correction was accurate. No changes needed.")
            
            else:
                print("\n⚠️ Invalid response. Skipping feedback logging.")
	
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()