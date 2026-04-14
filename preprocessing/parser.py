import json
import ast
from cleaner import clean_text
from language import detect_language

def parse_product_id(product_id):
    """
    Parses the product_id field, which may be a JSON string, a Python list string, or an empty string.
        - If it's a JSON string, it will be parsed into a Python list.
        - If it's a Python list string, it will be evaluated into a Python list.
        - If it's an empty string or cannot be parsed, it will return an empty list.

    Args:        product_id (str): The product_id field to parse.

    Returns:        list: A list of product IDs.
    """
    product_id = str(product_id).strip()
    product_id = product_id.encode().decode("unicode_escape")
    if product_id:
        try:
            return json.loads(product_id)
        except:
            try:
                return ast.literal_eval(product_id)
            except:
                return []
            
def process_row(row):
    """
    Process a single row of the DataFrame and return a dictionary with the relevant fields.
    Args:
        row (pd.Series): A row of the DataFrame.
    Returns:
        dict: A dictionary with the processed fields.
    """
    return {
        "doc_id": row["id"],
        "title": row["title"],
        "file_name": row["file_name"],
        "product_id": parse_product_id(row["product_id"]),
        "file_content": clean_text(row["file_content"]),
        "document_type": row["document_type"],
        "language": detect_language(row["file_content"])
    }