import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def serialize_full(row):
    """
    Serializes a row into a canonical dense representation.
    Output: [COUNTRY] india [NAME] some business [ADDRESS] 123 street
    """
    country = normalize_text(row.get('country', ''))
    name = normalize_text(row.get('business_name', ''))
    address = normalize_text(row.get('business_address', ''))
    
    parts = []
    if country:
        parts.append(f"[COUNTRY] {country}")
    if name:
        parts.append(f"[NAME] {name}")
    if address:
        parts.append(f"[ADDRESS] {address}")
        
    # Default to something if completely empty, to avoid empty embeddings
    if not parts:
        return "[EMPTY]"
    return " ".join(parts)

def serialize_name_only(row):
    """
    Serializes a row using only country and name.
    """
    country = normalize_text(row.get('country', ''))
    name = normalize_text(row.get('business_name', ''))
    
    parts = []
    if country:
        parts.append(f"[COUNTRY] {country}")
    if name:
        parts.append(f"[NAME] {name}")
        
    if not parts:
        return "[EMPTY]"
    return " ".join(parts)

def serialize_address_only(row):
    """
    Serializes a row using only country and address.
    """
    country = normalize_text(row.get('country', ''))
    address = normalize_text(row.get('business_address', ''))
    
    parts = []
    if country:
        parts.append(f"[COUNTRY] {country}")
    if address:
        parts.append(f"[ADDRESS] {address}")
        
    if not parts:
        return "[EMPTY]"
    return " ".join(parts)

def truncate_text(text, max_chars=1024):
    """
    Truncates text to a maximum number of characters to protect embedding context length.
    """
    if len(text) > max_chars:
        return text[:max_chars]
    return text

def extract_numerics(text):
    """
    Extracts purely numeric tokens or alphanumeric blocks from an address.
    """
    if not isinstance(text, str):
        return []
    # Find all contiguous blocks of digits
    numerics = re.findall(r'\b\d+\b', text)
    return numerics
