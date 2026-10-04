import re
import unicodedata

def normalize_vietnamese_text(text: str) -> str:
    """
    Defensively cleans and normalizes Vietnamese legal text.
    - Applies NFC Unicode normalization.
    - Removes zero-width and invisible formatting characters.
    - Standardizes line breaks while preserving paragraph structure.
    - Collapses multiple horizontal whitespaces into a single space.
    - Preserves all casing, punctuation, and legal identifiers.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Unicode Normalization (NFC is the standard for Vietnamese)
    # This prevents issues where 'ê' is represented as two different byte sequences.
    text = unicodedata.normalize('NFC', text)
    
    # 2. Remove Zero-width characters and BOM
    # \u200b to \u200d are zero-width spaces/joiners, \ufeff is Byte Order Mark
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # 3. Standardize line endings (Windows \r\n and old Mac \r to Linux \n)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 4. Clean horizontal whitespace (spaces, tabs) without removing newlines
    # Replaces 2 or more spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Strip trailing/leading spaces on each line to keep formatting crisp
    # (e.g., "Điều 5. \n" -> "Điều 5.\n")
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    
    # 6. Remove excess consecutive newlines (cap at 2 to preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 7. Final strip for the extreme start and end of the document
    return text.strip() 