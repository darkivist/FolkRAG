import os
import re
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
from pymarc import parse_xml_to_array
from datetime import datetime


def extract_afc_identifiers(url: str) -> List[str]:
    """Extract AFC identifiers from URLs in various formats."""
    patterns = set()

    url = url.lower().strip('/')
    url = url.replace('http://www.loc.gov/resource/', '')
    url = url.replace('https://www.loc.gov/resource/', '')
    url = url.replace('https://tile.loc.gov/storage-services/service/afc/', '')

    afc_pattern = re.search(r'(afc\d+)\.(afc\d+_\d+_(?:ms|sr|ph)\d+)', url)
    if afc_pattern:
        patterns.add(afc_pattern.group(2))
        patterns.add(f"{afc_pattern.group(1)}.{afc_pattern.group(2)}")

    parts = url.split('/')
    for part in parts:
        if 'afc' in part:
            patterns.add(part)
            if '_ms' in part or '_sr' in part or '_ph' in part:
                base_part = re.sub(r'\.(pdf|txt|mp3|wav)$', '', part)
                patterns.add(base_part)

                afc_match = re.match(r'(afc\d+)_(\d+_(?:ms|sr|ph)\d+)', base_part)
                if afc_match:
                    patterns.add(afc_match.group(0))
                    patterns.add(f"{afc_match.group(1)}.{afc_match.group(0)}")

    return list(patterns)

def clean_metadata_value(value):
    """Clean and convert metadata values to appropriate types."""
    if not value or value == '':
        return 'N/A'

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            try:
                cleaned_value = eval(value)
                if isinstance(cleaned_value, list):
                    return cleaned_value
            except:
                pass

        if value.lower() in ['n/a', 'none', '']:
            return 'N/A'

    return str(value)


def clean_metadata_dict(metadata: dict) -> dict:
    """Clean all values in a metadata dictionary."""
    cleaned = {}
    try:
        for key, value in metadata.items():
            cleaned[key] = clean_metadata_value(value)
    except Exception as e:
        print(f"Error in clean_metadata_dict: {e}")
        print(f"Problematic metadata: {metadata}")
        raise
    return cleaned


def extract_call_number(text: str) -> str:
    """Extract AFC call number from text."""
    if not text:
        return None
    pattern = r'\bAFC\s*\d{4}/\d{3}\b'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def process_digital_id(digital_id: str) -> List[str]:
    if not digital_id:
        return []

    patterns = set()
    base_id = digital_id.lower()

    # Remove common prefixes but preserve AFC suffixes
    base_id = base_id.replace('http://hdl.loc.gov/loc.afc/', '')
    base_id = re.sub(r'\.[^_]+$', '', base_id)  # Remove file extensions
    patterns.add(base_id)

    # Handle AFC-style IDs
    if 'afc' in base_id:
        # Split on dots first to handle afc2021007.afc2021007_002_ms01 format
        dot_parts = base_id.split('.')
        for part in dot_parts:
            if 'afc' in part:
                patterns.add(part)

                # If this part has ms/sr/ph suffix, preserve it
                if any(x in part for x in ['_ms', '_sr', '_ph']):
                    # Add both with and without the afc prefix
                    afc_match = re.match(r'(afc\d+)_(\d+_(?:ms|sr|ph)\d+)', part)
                    if afc_match:
                        patterns.add(afc_match.group(2))  # Add without AFC prefix
                        patterns.add(part)  # Add full version

    return list(patterns)

def process_resource_field(value: str, resource_type: str) -> List[str]:
    """Process a resource field value to generate potential matching patterns."""
    if not value:
        return []

    patterns = set()
    base_value = value.lower()

    # Clean the value
    base_value = base_value.replace('http://www.loc.gov/resource/', '')
    base_value = base_value.replace('http://www.loc.gov/item/', '')
    base_value = re.sub(r'\.[^_]+$', '', base_value)
    patterns.add(base_value)

    # Generate filename variations
    filename = os.path.basename(base_value)
    base_name = os.path.splitext(filename)[0]
    patterns.add(base_name)

    if resource_type == 'audio':
        patterns.add(f"{base_name}_en.txt")
        patterns.add(f"{base_name}_en_translation.txt")
        patterns.add(base_name.replace('.mp3', '').replace('.wav', ''))
    elif resource_type == 'pdf':
        patterns.add(f"{base_name}.txt")

    # Handle AFC patterns
    if 'afc' in base_name:
        parts = re.split(r'[._]', base_name)
        afc_parts = [p for p in parts if p.startswith('afc')]

        for afc_part in afc_parts:
            patterns.add(afc_part)

            if len(parts) > 1:
                idx = parts.index(afc_part)
                if idx < len(parts) - 1:
                    for i in range(idx + 1, len(parts)):
                        partial = '_'.join([afc_part] + parts[idx + 1:i + 1])
                        patterns.add(partial)

                        if any(x in partial for x in ['ms', 'sr', 'ph']):
                            base_pattern = re.sub(r'_(ms|sr|ph)\d+$', '', partial)
                            patterns.add(base_pattern)

    return list(patterns)


def parse_file_list_csv(file_path: str) -> Dict[str, str]:
    if not os.path.exists(file_path):
        return {}

    filename_to_id = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                doc_id = row['id']

                url_fields = ['source_url', 'resource_url', 'url', 'download']
                for field in url_fields:
                    if row.get(field):
                        identifiers = extract_afc_identifiers(row[field])
                        for identifier in identifiers:
                            filename_to_id[identifier] = doc_id
                            filename_to_id[f"{identifier}.txt"] = doc_id

    except Exception as e:
        return {}

    return filename_to_id


def parse_search_results_csv(file_path: str) -> Tuple[
    Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    if not os.path.exists(file_path):
        return {}, {}, {}, {}

    id_to_metadata = {}
    base_id_to_metadata = {}
    digital_id_to_metadata = {}
    resource_to_metadata = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata = {
                    'title': row['title'],
                    'contributors': [row.get(f'contributor.{i}', '') for i in range(3) if row.get(f'contributor.{i}')],
                    'date': row['date'],
                    'subjects': [row.get(f'subject.{i}', '') for i in range(5) if row.get(f'subject.{i}')],
                    'type': row.get('type.0', ''),
                    'language': row.get('language.0', ''),
                    'locations': [row.get(f'location.{i}', '') for i in range(3) if row.get(f'location.{i}')],
                    'original_format': row.get('original_format.0', ''),
                    'online_formats': [row.get(f'online_format.{i}', '') for i in range(2) if
                                       row.get(f'online_format.{i}')],
                    'description': row.get('description', ''),
                    'rights': row.get('rights', ''),
                    'collection': row.get('collection', ''),
                    'timestamp': row.get('timestamp', ''),
                    'created_published': row.get('item.created_published.0', ''),
                    'notes': [row.get(f'item.notes.{i}', '') for i in range(2) if row.get(f'item.notes.{i}')],
                    'url': row.get('url', '')
                }

                id_to_metadata[row['id']] = metadata

                for i in range(7):
                    aka_value = row.get(f'aka.{i}')
                    if aka_value:
                        identifiers = extract_afc_identifiers(aka_value)
                        for identifier in identifiers:
                            base_id_to_metadata[identifier] = metadata
                            base_id_to_metadata[f"{identifier}.txt"] = metadata

                for i in range(5):
                    digital_id = row.get(f'item.digital_id.{i}')
                    if digital_id:
                        identifiers = extract_afc_identifiers(digital_id)
                        for identifier in identifiers:
                            digital_id_to_metadata[identifier] = metadata
                            digital_id_to_metadata[f"{identifier}.txt"] = metadata

                for i in range(5):
                    resource_url = row.get(f'resources.{i}.url')
                    if resource_url:
                        identifiers = extract_afc_identifiers(resource_url)
                        for identifier in identifiers:
                            resource_to_metadata[identifier] = metadata
                            resource_to_metadata[f"{identifier}.txt"] = metadata

    except Exception as e:
        return {}, {}, {}, {}

    return id_to_metadata, base_id_to_metadata, digital_id_to_metadata, resource_to_metadata

def match_base_id(filename: str, base_id_to_metadata: Dict[str, Dict],
                  digital_id_to_metadata: Dict[str, Dict],
                  resource_to_metadata: Dict[str, Dict]) -> Dict:
    """Match a file to its metadata using all available mappings."""
    print(f"\n=== Attempting to match: {filename} ===")

    # Clean the filename
    base_name = filename.lower()
    if '_en_translation' in base_name or '_en.' in base_name:
        base_name = re.sub(r'_[a-z]{2}_en_translation\..*$', '', base_name)
        base_name = re.sub(r'_en\..*$', '', base_name)
    else:
        base_name = re.sub(r'\.txt$', '', base_name)

    print(f"1. Stripped filename: {base_name}")

    # Try each mapping in order
    for mapping_name, mapping in [
        ("base ID", base_id_to_metadata),
        ("digital ID", digital_id_to_metadata),
        ("resource", resource_to_metadata)
    ]:
        print(f"2. Trying {mapping_name} match: {base_name}")
        if base_name in mapping:
            return mapping[base_name]

        # Try with AFC format
        afc_prefix_match = re.match(r'(afc\d+)', base_name)
        if afc_prefix_match:
            afc_prefix = afc_prefix_match.group(1)
            afc_format = f"{afc_prefix}.{base_name}"
            print(f"3. Trying AFC format in {mapping_name} mapping: {afc_format}")
            if afc_format in mapping:
                return mapping[afc_format]

        # Try without suffix as last resort
        main_parts = re.match(r'(afc\d+_\d+)', base_name)
        if main_parts and main_parts.group(1) != base_name:
            print(f"4. Trying main identifier in {mapping_name} mapping: {main_parts.group(1)}")
            if main_parts.group(1) in mapping:
                return mapping[main_parts.group(1)]

    print("\nNo matches found. Available similar patterns:")
    prefix = base_name[:8]
    all_mappings = {**base_id_to_metadata, **digital_id_to_metadata, **resource_to_metadata}
    matching_patterns = [pat for pat in all_mappings.keys() if pat.startswith(prefix)]

    for pattern in matching_patterns[:5]:
        print(f"  - {pattern}")
    if len(matching_patterns) > 5:
        print(f"  ... and {len(matching_patterns) - 5} more")

    return None


def process_metadata(data_dir: str) -> Dict[str, Dict]:
    """Process metadata from all sources and combine them."""
    loc_data_dir = os.path.join(data_dir, 'loc_dot_gov_data')
    txt_dir = os.path.join(data_dir, 'txt')
    transcripts_dir = os.path.join(data_dir, 'transcripts')
    ocr_dir = os.path.join(data_dir, 'pdf', 'txtConversion')

    all_metadata = {}
    error_log = []
    missing_metadata_log = {}

    # Add debug counters
    debug_counts = {
        'total_files': 0,
        'processed_files': {
            'text': 0,
            'transcript': 0,
            'OCR': 0
        },
        'successful_matches': {
            'text': 0,
            'transcript': 0,
            'OCR': 0
        },
        'failed_matches': {
            'text': 0,
            'transcript': 0,
            'OCR': 0
        }
    }

    for collection_name in os.listdir(loc_data_dir):
        collection_dir = os.path.join(loc_data_dir, collection_name)
        if os.path.isdir(collection_dir):
            file_list_path = os.path.join(collection_dir, 'file_list.csv')
            search_results_path = os.path.join(collection_dir, 'search_results.csv')

            filename_to_id = parse_file_list_csv(file_list_path)
            id_to_metadata, base_id_to_metadata, digital_id_to_metadata, resource_to_metadata = parse_search_results_csv(
                search_results_path)

            for directory in [txt_dir, transcripts_dir, ocr_dir]:
                if not os.path.exists(directory):
                    continue

                dir_type = {
                    txt_dir: 'text',
                    transcripts_dir: 'transcript',
                    ocr_dir: 'OCR'
                }[directory]

                for filename in os.listdir(directory):
                    if filename.endswith('.txt'):
                        debug_counts['total_files'] += 1
                        debug_counts['processed_files'][dir_type] += 1

                        try:
                            metadata = None
                            matched_id = filename_to_id.get(filename)
                            if matched_id and matched_id in id_to_metadata:
                                metadata = id_to_metadata[matched_id].copy()

                            if not metadata:
                                metadata = match_base_id(filename, base_id_to_metadata,
                                                         digital_id_to_metadata, resource_to_metadata)

                            if metadata:
                                metadata['dir_type'] = dir_type
                                metadata['filename'] = filename
                                metadata = clean_metadata_dict(metadata)
                                all_metadata[filename] = metadata
                                debug_counts['successful_matches'][dir_type] += 1
                            else:
                                debug_counts['failed_matches'][dir_type] += 1
                                if filename not in missing_metadata_log:
                                    missing_metadata_log[filename] = {
                                        'filename': filename,
                                        'directory_type': dir_type,
                                        'tried_file_list': matched_id is not None,
                                        'tried_alternative_matching': True,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                error_msg = f"No metadata found for {filename}"
                                error_log.append(error_msg)
                                print(f"Warning: {error_msg}")

                        except Exception as e:
                            error_msg = f"Error processing file {filename}: {str(e)}"
                            error_log.append(error_msg)
                            debug_counts['failed_matches'][dir_type] += 1

    # Print detailed debug counts
    #print("\n=== Debug Counts ===")
    #print(f"Total files found: {debug_counts['total_files']}")
    #print("\nProcessed files by directory:")
    #for dir_type, count in debug_counts['processed_files'].items():
        #print(f"  {dir_type}: {count}")
    print("\nSuccessful matches by directory:")
    for dir_type, count in debug_counts['successful_matches'].items():
        print(f"  {dir_type}: {count}")
    #print("\nFailed matches by directory:")
    #for dir_type, count in debug_counts['failed_matches'].items():
        #print(f"  {dir_type}: {count}")

    # Write detailed missing metadata report
    missing_metadata_path = os.path.join(data_dir, 'missing_metadata_report.txt')
    with open(missing_metadata_path, 'w') as f:
        f.write("=== Missing Metadata Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total files processed: {len(all_metadata) + len(missing_metadata_log)}\n")
        f.write(f"Files with metadata: {len(all_metadata)}\n")
        f.write(f"Files missing metadata: {len(missing_metadata_log)}\n\n")

        dir_stats = {}
        for entry in missing_metadata_log.values():
            dir_type = entry['directory_type']
            dir_stats[dir_type] = dir_stats.get(dir_type, 0) + 1

        f.write("Missing Metadata by Directory Type:\n")
        f.write("-" * 50 + "\n")
        for dir_type, count in sorted(dir_stats.items()):
            f.write(f"{dir_type}: {count} files\n")

        f.write(f"\nDetailed Missing Entries:\n")
        for entry in sorted(missing_metadata_log.values(), key=lambda x: x['filename']):
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"File: {entry['filename']}\n")
            f.write(f"Directory Type: {entry['directory_type']}\n")
            f.write(f"Tried file_list.csv: {entry['tried_file_list']}\n")
            f.write(f"Tried alternative matching: {entry['tried_alternative_matching']}\n")
            f.write(f"Processed: {entry['timestamp']}\n")

        # Get directory type breakdown
        dir_stats = {}
        for entry in missing_metadata_log.values():
            dir_type = entry['directory_type']
            dir_stats[dir_type] = dir_stats.get(dir_type, 0) + 1

        print("\n=== Metadata Processing Complete ===")
        print(f"Successfully processed metadata for {len(all_metadata)} files")
        #print(f"Failed to find metadata for {len(missing_metadata_log)} files")
        #print("\nBreakdown of missing metadata by directory type:")
        #for dir_type, count in debug_counts['failed_matches'].items():
            #print(f"  {dir_type}: {count} files")
        print(f"\nDetailed report written to: {missing_metadata_path}")

        return all_metadata