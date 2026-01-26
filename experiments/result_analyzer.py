"""
Result analyzer for GlobalHealthAtlas
"""
import json
import os
from typing import List, Dict, Any
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font


def analyze_results(input_file: str = './outputs/scored.json', output_file: str = './outputs/scored_detailed.xlsx'):
    """
    Analyze scoring results and export to Excel
    """
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    # Filter items with scores
    valid_items = [
        item for item in data 
        if 'scores' in item and item['scores'] is not None and item['scores']
    ]
    print(f"Items with scores: {len(valid_items)}")
    
    # Aggregate scores by label, domain, and language
    dimensions = [
        'Accuracy', 
        'Reasoning', 
        'Completeness', 
        'Consensus Alignment', 
        'Terminology Norms', 
        'Insightfulness'
    ]
    
    # Initialize data structure
    data_structure = defaultdict(lambda: {'sums': [0.0] * 6, 'count': 0})
    
    for item in valid_items:
        scores = item['scores']
        label = item.get('label', 'unknown')
        domain = item.get('domain', 'unknown')
        language = item.get('language', 'unknown')
        
        key = (label, domain, language)
        
        # Extract scores
        for i, dim in enumerate(dimensions):
            if dim in scores and 'score' in scores[dim]:
                data_structure[key]['sums'][i] += scores[dim]['score']
        data_structure[key]['count'] += 1
    
    aggregated = dict(data_structure)
    
    # Calculate averages
    result = {}
    for key, stats in aggregated.items():
        count = stats['count']
        avgs = [s / count if count > 0 else 0 for s in stats['sums']]
        result[key] = {
            'averages': avgs,
            'count': count
        }
    
    # Export to Excel
    export_to_excel(output_file, result, dimensions)
    
    # Print summary
    print_summary(result, dimensions)


def export_to_excel(file_path: str, aggregated_data: Dict[tuple, Dict[str, Any]], dimensions: List[str]):
    """
    Export aggregated scores to Excel
    
    Args:
        file_path: Output Excel file path
        aggregated_data: Aggregated score data with averages
        dimensions: List of dimension names
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Scores Summary"
    
    # Write header
    header = ['label', 'domain', 'language', 'count']
    header.extend(dimensions)
    ws.append(header)
    
    # Style header
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    
    # Write data rows
    sorted_keys = sorted(aggregated_data.keys(), key=lambda x: (x[0], x[1], x[2]))
    
    for key in sorted_keys:
        label, domain, language = key
        data = aggregated_data[key]
        count = data['count']
        avgs = [f"{x:.4f}" for x in data['averages']]
        
        row = [label, domain, language, count]
        row.extend(avgs)
        ws.append(row)
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 30)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Save workbook
    wb.save(file_path)
    print(f"\nExcel file saved to: {file_path}")


def print_summary(aggregated_data: Dict[tuple, Dict[str, Any]], dimensions: List[str]):
    """
    Print summary statistics
    
    Args:
        aggregated_data: Aggregated score data
        dimensions: List of dimension names
    """
    sorted_keys = sorted(aggregated_data.keys(), key=lambda x: (x[0], x[1], x[2]))
    
    print(f"\n{'='*60}")
    print("SCORE SUMMARY")
    print(f"{'='*60}")
    
    for key in sorted_keys:
        label, domain, language = key
        data = aggregated_data[key]
        count = data['count']
        avgs = [f"{x:.4f}" for x in data['averages']]
        
        print(f"{label}, {domain}, {language}: "
              f"count={count}, averages={avgs}")
    
    print(f"\nTotal combinations: {len(sorted_keys)}")
    print(f"{'='*60}\n")