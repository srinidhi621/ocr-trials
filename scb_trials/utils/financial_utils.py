"""
Financial Value Utilities
Helper functions for extracting and validating financial values.
"""

import re
from typing import List, Dict, Optional, Tuple


# Currency symbols and codes
CURRENCY_SYMBOLS = {
    '$': 'USD',
    '£': 'GBP',
    '€': 'EUR',
    '¥': 'JPY',
    '₹': 'INR',
    'Rs': 'INR',
    'Rs.': 'INR',
}

CURRENCY_CODES = ['INR', 'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'SGD', 'CHF', 'HKD']

# Unit multipliers
UNIT_MULTIPLIERS = {
    'million': 1_000_000,
    'mn': 1_000_000,
    'billion': 1_000_000_000,
    'bn': 1_000_000_000,
    'trillion': 1_000_000_000_000,
    'crore': 10_000_000,
    'cr': 10_000_000,
    'lakh': 100_000,
    'lac': 100_000,
    'lakhs': 100_000,
    'lacs': 100_000,
    'thousand': 1_000,
    'k': 1_000,
}


def extract_financial_values(text: str) -> List[Dict[str, str]]:
    """
    Extract all financial values from text.
    
    Args:
        text: Text to search
        
    Returns:
        List of dictionaries with value, type, and original string
    """
    results = []
    
    # Pattern 1: Currency symbol/code + number
    pattern1 = r'(?:' + '|'.join(re.escape(s) for s in CURRENCY_SYMBOLS.keys()) + r'|' + '|'.join(CURRENCY_CODES) + r')\s*[\d,]+(?:\.\d+)?'
    
    # Pattern 2: Number + currency code
    pattern2 = r'[\d,]+(?:\.\d+)?\s*(?:' + '|'.join(CURRENCY_CODES) + ')'
    
    # Pattern 3: Number + unit (Million, Crore, etc.)
    units = '|'.join(UNIT_MULTIPLIERS.keys())
    pattern3 = r'[\d,]+(?:\.\d+)?\s*(?:' + units + ')'
    
    # Pattern 4: Percentage
    pattern4 = r'[\d,]+(?:\.\d+)?\s*%'
    
    patterns = [
        (pattern1, 'currency'),
        (pattern2, 'currency'),
        (pattern3, 'amount_with_unit'),
        (pattern4, 'percentage'),
    ]
    
    seen = set()
    for pattern, value_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group().strip()
            if value not in seen:
                seen.add(value)
                results.append({
                    'value': value,
                    'type': value_type,
                    'start': match.start(),
                    'end': match.end()
                })
    
    return results


def normalize_financial_value(value: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Normalize a financial value to a numeric amount and currency.
    
    Args:
        value: Financial value string
        
    Returns:
        Tuple of (numeric_value, currency_code) or (None, None) if parsing fails
    """
    if not value:
        return None, None
    
    # Clean the value
    cleaned = value.strip()
    
    # Extract currency
    currency = None
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in cleaned:
            currency = code
            cleaned = cleaned.replace(symbol, '')
            break
    
    if currency is None:
        for code in CURRENCY_CODES:
            if code.upper() in cleaned.upper():
                currency = code.upper()
                cleaned = re.sub(code, '', cleaned, flags=re.IGNORECASE)
                break
    
    # Extract unit multiplier
    multiplier = 1
    cleaned_lower = cleaned.lower()
    for unit, mult in UNIT_MULTIPLIERS.items():
        if unit in cleaned_lower:
            multiplier = mult
            cleaned = re.sub(unit, '', cleaned, flags=re.IGNORECASE)
            break
    
    # Extract numeric value
    cleaned = cleaned.strip()
    cleaned = cleaned.replace(',', '')
    cleaned = cleaned.replace(' ', '')
    
    # Handle percentage
    is_percentage = '%' in cleaned
    cleaned = cleaned.replace('%', '')
    
    try:
        numeric = float(cleaned) * multiplier
        if is_percentage:
            currency = '%'
        return numeric, currency
    except ValueError:
        return None, None


def validate_financial_format(value: str) -> bool:
    """
    Validate that a financial value is properly formatted.
    
    Args:
        value: Financial value string
        
    Returns:
        True if valid format, False otherwise
    """
    if not value:
        return False
    
    # Check for valid patterns
    patterns = [
        r'^[A-Z]{3}\s*[\d,]+(?:\.\d{2})?$',  # INR 500,000.00
        r'^[\d,]+(?:\.\d{2})?\s*[A-Z]{3}$',  # 500,000.00 INR
        r'^[$£€¥₹]\s*[\d,]+(?:\.\d{2})?$',  # $500,000.00
        r'^[\d,]+(?:\.\d+)?\s*(?:Million|Billion|Crore|Lakh)s?$',  # 500 Million
        r'^[\d,]+(?:\.\d+)?\s*%$',  # 15.5%
    ]
    
    for pattern in patterns:
        if re.match(pattern, value.strip(), re.IGNORECASE):
            return True
    
    return False


def format_financial_value(
    amount: float,
    currency: str = 'INR',
    decimals: int = 2,
    use_unit: bool = False
) -> str:
    """
    Format a numeric amount as a financial value string.
    
    Args:
        amount: Numeric amount
        currency: Currency code
        decimals: Decimal places
        use_unit: Whether to use unit abbreviations (Million, Crore, etc.)
        
    Returns:
        Formatted financial value string
    """
    if use_unit:
        # Determine appropriate unit
        if amount >= 1_000_000_000:
            formatted = f"{amount / 1_000_000_000:,.{decimals}f} Billion"
        elif amount >= 10_000_000:
            formatted = f"{amount / 10_000_000:,.{decimals}f} Crore"
        elif amount >= 1_000_000:
            formatted = f"{amount / 1_000_000:,.{decimals}f} Million"
        elif amount >= 100_000:
            formatted = f"{amount / 100_000:,.{decimals}f} Lakh"
        else:
            formatted = f"{amount:,.{decimals}f}"
    else:
        formatted = f"{amount:,.{decimals}f}"
    
    if currency == '%':
        return f"{formatted}%"
    
    return f"{currency} {formatted}"


def compare_financial_values(value1: str, value2: str) -> bool:
    """
    Compare two financial values for equality.
    
    Handles different formats (e.g., "INR 50 Crore" vs "INR 500,000,000").
    
    Args:
        value1: First financial value
        value2: Second financial value
        
    Returns:
        True if values are equivalent, False otherwise
    """
    num1, curr1 = normalize_financial_value(value1)
    num2, curr2 = normalize_financial_value(value2)
    
    if num1 is None or num2 is None:
        return False
    
    # Check currency match (if both have currency)
    if curr1 and curr2 and curr1 != curr2:
        return False
    
    # Compare numeric values with tolerance for floating point
    tolerance = max(abs(num1), abs(num2)) * 0.0001  # 0.01% tolerance
    return abs(num1 - num2) <= tolerance
