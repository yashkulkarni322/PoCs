import fitz
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Union
from utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting text from TXT {txt_path}: {str(e)}")
        raise


def extract_text_from_html(html_path: str) -> str:
    """Extract text from HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        logger.error(f"Error extracting text from HTML {html_path}: {str(e)}")
        raise


def format_csv_data(df: pd.DataFrame) -> str:
    """Format CSV DataFrame to structured text"""
    text_parts = []
    
    headers = " | ".join(str(col) for col in df.columns)
    text_parts.append(f"CSV Headers: {headers}")
    text_parts.append("-" * len(headers))
    
    for idx, row in df.iterrows():
        row_text = " | ".join(str(value) if pd.notna(value) else "" for value in row)
        text_parts.append(f"Row {idx + 1}: {row_text}")
    
    text_parts.append(f"\nCSV Summary: {len(df)} rows, {len(df.columns)} columns")
    return "\n".join(text_parts)


def extract_text_from_csv(csv_path: str) -> str:
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        return format_csv_data(df)
    except Exception as e:
        logger.error(f"Error extracting text from CSV {csv_path}: {str(e)}")
        raise


def format_excel_sheet(df: pd.DataFrame, sheet_name: str) -> str:
    """Format Excel sheet data to text"""
    if df.empty:
        return f"\n=== Sheet: {sheet_name} ===\nEmpty sheet"
    
    sheet_text = [f"\n=== Sheet: {sheet_name} ===\n"]
    
    headers = " | ".join(str(col) for col in df.columns)
    sheet_text.append(f"Headers: {headers}")
    sheet_text.append("-" * len(headers))
    
    for idx, row in df.iterrows():
        row_text = " | ".join(str(value) if pd.notna(value) else "" for value in row)
        sheet_text.append(f"Row {idx + 1}: {row_text}")
    
    return "\n".join(sheet_text)


def extract_text_from_excel(excel_path: str) -> str:
    """Extract text from Excel file"""
    try:
        xl_file = pd.ExcelFile(excel_path)
        all_text = []
        
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            sheet_text = format_excel_sheet(df, sheet_name)
            all_text.append(sheet_text)
        
        return "\n".join(all_text)
    except Exception as e:
        logger.error(f"Error extracting text from Excel {excel_path}: {str(e)}")
        raise


def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase"""
    return Path(file_path).suffix.lower()


def extract_text_by_type(file_path: str) -> str:
    """Extract text from file based on extension"""
    extension = get_file_extension(file_path)
    
    extractors = {
        '.pdf': extract_text_from_pdf,
        '.txt': extract_text_from_txt,
        '.html': extract_text_from_html,
        '.htm': extract_text_from_html,
        '.csv': extract_text_from_csv,
        '.xlsx': extract_text_from_excel,
        '.xls': extract_text_from_excel,
    }
    
    extractor = extractors.get(extension)
    if not extractor:
        raise ValueError(f"Unsupported file format: {extension}")
    
    return extractor(file_path)