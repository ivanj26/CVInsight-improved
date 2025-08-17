from datetime import datetime
from .. import config
from .. import constants

def parse_date(date_str, default_day="01", default_month="01"):
    """
    Parse a date string into a datetime object.
    
    Args:
        date_str: The date string to parse.
        default_day: The default day to use if not provided.
        default_month: The default month to use if not provided.
        
    Returns:
        A datetime object.
    """
    if not date_str:
        return None
        
    try:
        return datetime.strptime(date_str, config.DATE_FORMAT)
    except ValueError:
        # Attempt to handle various date formats
        pass
    
    return None

def calculate_experience(oldest_date_str, newest_date_str):
    """
    Calculate total years of experience between oldest_working_date and newest_working_date.
    
    Args:
        oldest_date_str: The oldest working date string in "dd/mm/yyyy" format.
        newest_date_str: The newest working date string in "dd/mm/yyyy" format.
        
    Returns:
        A string representing the total experience in "X Years Y Months" format.
    """
    if not oldest_date_str or not newest_date_str:
        return "0Y 0M"
    
    try:
        oldest_date = datetime.strptime(oldest_date_str, config.DATE_FORMAT)
        newest_date = datetime.strptime(newest_date_str, config.DATE_FORMAT)
    except ValueError:
        # If the date format is invalid, return default experience.
        return "0Y 0M"
    
    # Calculate total months of difference.
    total_months = (newest_date.year - oldest_date.year) * 12 + (newest_date.month - oldest_date.month)
    
    # Adjust if the day of the month in the newest date is less than the oldest date.
    if newest_date.day < oldest_date.day:
        total_months -= 1
    
    years = total_months // 12
    months = total_months % 12
    
    return f"{years}Y {months}M" 