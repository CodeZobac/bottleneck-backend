import pandas as pd
import re
import logging
import os
from typing import Dict, List, Any
from errors import ComponentDataError, FileNotFoundError, DataFormatError

logger = logging.getLogger(__name__)

def clean_component_name(name: str) -> str:
    """Clean component names by removing duplicated words and normalizing format."""
    # Split the name into words
    words = name.split()
    
    # Remove duplicate consecutive words 
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            cleaned_words.append(word)
    
    return " ".join(cleaned_words)

def categorize_gpus(gpu_data: list) -> List[Dict[str, Any]]:
    """Categorize GPUs by brand/series for the frontend, showing only unique core models."""
    
    categories = {
        "RTX": {"id": 1, "name": "NVIDIA RTX", "models": []},
        "GTX": {"id": 2, "name": "NVIDIA GTX", "models": []},
        "AMD": {"id": 3, "name": "AMD Radeon", "models": []},
        "Intel": {"id": 4, "name": "Intel", "models": []},
        "Other": {"id": 5, "name": "Other GPUs", "models": []}
    }
    
    model_id = 100
    # Track models we've already added to avoid duplicates
    added_models = set()
    
    # Common GPU model patterns
    rtx_pattern = r'(RTX\s+\d+\w*(?:\s+\w+)?)'
    gtx_pattern = r'(GTX\s+\d+\w*(?:\s+\w+)?)'
    amd_patterns = [
        r'(RX\s+\d+\w*(?:\s+\w+)?)',
        r'(Radeon\s+\w+\s+\d+\w*)',
        r'(Vega\s+\d+\w*)'
    ]
    intel_pattern = r'(Intel\s+\w+\s+\d+\w*)'
    
    for gpu in gpu_data:
        # Extract the core model based on patterns
        core_model = None
        category = None
        
        if "RTX" in gpu:
            match = re.search(rtx_pattern, gpu, re.IGNORECASE)
            if match:
                core_model = match.group(1)
                category = "RTX"
        elif "GTX" in gpu:
            match = re.search(gtx_pattern, gpu, re.IGNORECASE)
            if match:
                core_model = match.group(1)
                category = "GTX"
        elif any(amd_term in gpu for amd_term in ["AMD", "Radeon", "RX", "Vega"]):
            for pattern in amd_patterns:
                match = re.search(pattern, gpu, re.IGNORECASE)
                if match:
                    core_model = match.group(1)
                    category = "AMD"
                    break
        elif "Intel" in gpu:
            match = re.search(intel_pattern, gpu, re.IGNORECASE)
            if match:
                core_model = match.group(1)
                category = "Intel"
        
        # If we couldn't extract a core model, use the full name
        if not core_model:
            core_model = clean_component_name(gpu)
            category = "Other"
        
        # Normalize the model name and check if we've already added it
        core_model = core_model.strip()
        if (core_model.lower() not in added_models):
            categories[category]["models"].append({
                "id": model_id,
                "name": core_model
            })
            added_models.add(core_model.lower())
            model_id += 1
    
    # Filter out empty categories and convert to list
    result = [cat for cat in categories.values() if cat["models"]]
    
    # Sort models by name within each category
    for category in result:
        category["models"].sort(key=lambda x: x["name"])
    
    return result

def categorize_cpus(cpu_data: list) -> List[Dict[str, Any]]:
    """Categorize CPUs by brand/series for the frontend."""
    
    categories = {
        "Intel Core i9": {"id": 10, "name": "Intel Core i9", "models": []},
        "Intel Core i7": {"id": 11, "name": "Intel Core i7", "models": []},
        "Intel Core i5": {"id": 12, "name": "Intel Core i5", "models": []},
        "Intel Core i3": {"id": 13, "name": "Intel Core i3", "models": []},
        "AMD Ryzen 9": {"id": 14, "name": "AMD Ryzen 9", "models": []},
        "AMD Ryzen 7": {"id": 15, "name": "AMD Ryzen 7", "models": []},
        "AMD Ryzen 5": {"id": 16, "name": "AMD Ryzen 5", "models": []},
        "AMD Ryzen 3": {"id": 17, "name": "AMD Ryzen 3", "models": []},
        "Other Intel": {"id": 18, "name": "Other Intel CPUs", "models": []},
        "Other AMD": {"id": 19, "name": "Other AMD CPUs", "models": []},
        "Other": {"id": 20, "name": "Other CPUs", "models": []}
    }
    
    model_id = 1000
    
    for cpu in cpu_data:
        clean_name = clean_component_name(cpu)
        
        # Determine category
        if "Intel" in clean_name:
            if "Core i9" in clean_name:
                category = "Intel Core i9"
            elif "Core i7" in clean_name:
                category = "Intel Core i7"
            elif "Core i5" in clean_name:
                category = "Intel Core i5"
            elif "Core i3" in clean_name:
                category = "Intel Core i3"
            else:
                category = "Other Intel"
        elif "AMD" in clean_name:
            if "Ryzen 9" in clean_name:
                category = "AMD Ryzen 9"
            elif "Ryzen 7" in clean_name:
                category = "AMD Ryzen 7"
            elif "Ryzen 5" in clean_name:
                category = "AMD Ryzen 5"
            elif "Ryzen 3" in clean_name:
                category = "AMD Ryzen 3"
            else:
                category = "Other AMD"
        else:
            category = "Other"
        
        # Add to category with unique ID
        categories[category]["models"].append({
            "id": model_id,
            "name": clean_name
        })
        model_id += 1
    
    # Filter out empty categories and convert to list
    result = [cat for cat in categories.values() if cat["models"]]
    
    # Sort models by name within each category
    for category in result:
        category["models"].sort(key=lambda x: x["name"])
    
    return result

def generate_ram_categories() -> List[Dict[str, Any]]:
    """Generate RAM categories with common speeds."""
    categories = {
        "DDR5": {"id": 30, "name": "DDR5", "models": []},
        "DDR4": {"id": 31, "name": "DDR4", "models": []},
        "DDR3": {"id": 32, "name": "DDR3", "models": []}
    }
    
    model_id = 2000
    
    # DDR5 speeds
    for speed in range(4800, 8801, 400):
        categories["DDR5"]["models"].append({
            "id": model_id,
            "name": f"DDR5 {speed}MHz"
        })
        model_id += 1
    
    # DDR4 speeds
    for speed in range(2133, 4801, 267):
        categories["DDR4"]["models"].append({
            "id": model_id,
            "name": f"DDR4 {speed}MHz"
        })
        model_id += 1
    
    # DDR3 speeds
    for speed in range(1333, 2401, 267):
        categories["DDR3"]["models"].append({
            "id": model_id,
            "name": f"DDR3 {speed}MHz"
        })
        model_id += 1
    
    return list(categories.values())

def categorize_ram(ram_path: str) -> List[Dict[str, Any]]:
    """Categorize RAM data from CSV by DDR generation."""
    
    categories = {
        "DDR5": {"id": 30, "name": "DDR5", "models": []},
        "DDR4": {"id": 31, "name": "DDR4", "models": []},
        "DDR3": {"id": 32, "name": "DDR3", "models": []},
        "Other": {"id": 33, "name": "Other RAM", "models": []}
    }
    
    model_id = 2000
    added_models = set()
    
    try:
        # Check if file exists
        if not os.path.exists(ram_path):
            logger.warning(f"RAM data file not found: {ram_path}, generating default values")
            return generate_ram_categories()
        
        # Load RAM data
        ram_df = pd.read_csv(ram_path)
        
        # Verify required column exists
        if 'Model' not in ram_df.columns:
            logger.warning("Missing 'Model' column in RAM data, generating default values")
            return generate_ram_categories()
        
        # Extract DDR type and speed from model description
        ddr_pattern = r'(DDR\d)\s+(\d+)'
        
        for _, row in ram_df.iterrows():
            try:
                model_desc = str(row['Model'])
                match = re.search(ddr_pattern, model_desc, re.IGNORECASE)
                
                if match:
                    ddr_type = match.group(1)
                    speed = match.group(2)
                    model_name = f"{ddr_type} {speed}MHz"
                    
                    # Determine category
                    if ddr_type.upper() in categories:
                        category = ddr_type.upper()
                    else:
                        category = "Other"
                    
                    # Add to category if not already added
                    if model_name.lower() not in added_models:
                        categories[category]["models"].append({
                            "id": model_id,
                            "name": model_name
                        })
                        added_models.add(model_name.lower())
                        model_id += 1
            except Exception as e:
                logger.warning(f"Error processing RAM row: {str(e)}, row data: {row}")
                # Continue processing other rows
        
        # If no RAM models were processed, fall back to default values
        if not any(cat["models"] for cat in categories.values()):
            logger.warning("No RAM models extracted from data, generating default values")
            return generate_ram_categories()
        
        # Filter out empty categories and convert to list
        result = [cat for cat in categories.values() if cat["models"]]
        
        # Sort models by numeric speed within each category
        for category in result:
            category["models"].sort(key=lambda x: int(re.search(r'(\d+)MHz', x["name"]).group(1)))
        
        return result
        
    except Exception as e:
        logger.error(f"Error categorizing RAM data: {str(e)}", exc_info=True)
        # Fall back to default values in case of error
        return generate_ram_categories()

def get_categorized_components(cpu_path: str, gpu_path: str, ram_path: str = "./data/RAM_UserBenchmarks.csv") -> Dict[str, Any]:
    """Get categorized components for the frontend."""
    try:
        # Check if CPU and GPU files exist
        for path, component in [(cpu_path, "CPU"), (gpu_path, "GPU")]:
            if not os.path.exists(path):
                logger.error(f"{component} data file not found: {path}")
                raise FileNotFoundError(f"{component} data file not found: {path}")
        
        # Load data
        try:
            cpu_df = pd.read_csv(cpu_path)
            gpu_df = pd.read_csv(gpu_path)
        except Exception as e:
            logger.error(f"Failed to read CSV files: {str(e)}")
            raise DataFormatError(f"Failed to read CSV files: {str(e)}")
        
        # Verify required columns exist
        for df, name, required_cols in [
            (cpu_df, "CPU", ['Brand', 'Model']), 
            (gpu_df, "GPU", ['Brand', 'Model'])
        ]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {name} data: {missing_cols}")
                raise DataFormatError(f"Missing required columns in {name} data: {missing_cols}")
        
        # Process CPU data
        cpu_components = []
        for _, row in cpu_df.iterrows():
            try:
                component_name = f"{row['Brand']} {row['Model']}"
                cpu_components.append(component_name)
            except Exception as e:
                logger.warning(f"Error processing CPU row: {str(e)}, row data: {row}")
                # Continue processing other rows
        
        # Process GPU data
        gpu_components = []
        for _, row in gpu_df.iterrows():
            try:
                component_name = f"{row['Brand']} {row['Model']}"
                gpu_components.append(component_name)
            except Exception as e:
                logger.warning(f"Error processing GPU row: {str(e)}, row data: {row}")
                # Continue processing other rows
        
        # Return empty lists if no components were processed
        if not cpu_components:
            logger.warning("No CPU components were processed successfully")
        if not gpu_components:
            logger.warning("No GPU components were processed successfully")
        
        # Get RAM categories from CSV - FIX: Only call the function once
        ram_categories = categorize_ram(ram_path)
        
        # Log RAM categories to help diagnose issues
        ram_model_count = sum(len(cat['models']) for cat in ram_categories)
        logger.info(f"RAM categories processed: {len(ram_categories)} categories with {ram_model_count} total models")
        for category in ram_categories:
            logger.info(f"  - {category['name']}: {len(category['models'])} models")
        
        # Categorize components with field names that match the API expectations
        result = {
            "cpus": categorize_cpus(cpu_components),
            "gpus": categorize_gpus(gpu_components),
            "rams": ram_categories  # FIX: Use the already processed RAM categories
        }
        
        # Add additional logging for debugging
        logger.info(f"API response structure: {list(result.keys())}")
        logger.info(f"CPU categories: {len(result['cpus'])}, GPU categories: {len(result['gpus'])}, RAM categories: {len(result['rams'])}")
        
        return result
    except Exception as e:
        logger.error(f"Error getting component data: {str(e)}", exc_info=True)
        # Re-raise as a ComponentDataError to be handled by the API
        if not isinstance(e, (FileNotFoundError, DataFormatError)):
            raise ComponentDataError(f"Unexpected error processing component data: {str(e)}")
        raise
