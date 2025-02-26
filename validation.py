import pandas as pd
import logging

# Get the logger
logger = logging.getLogger(__name__)

class BenchmarkValidator:
	def __init__(self, cpu_csv, gpu_csv, ram_csv):
		self.cpu_data = pd.read_csv(cpu_csv)
		self.gpu_data = pd.read_csv(gpu_csv)
		self.ram_data = pd.read_csv(ram_csv)
		logger.info(f"Loaded benchmark data - CPU: {len(self.cpu_data)} rows, GPU: {len(self.gpu_data)} rows, RAM: {len(self.ram_data)} rows")

	def find_benchmark(self, component_type, model_name):
		"""Find benchmark score for a given component model."""
		logger.info(f"Searching for benchmark - Type: {component_type}, Model: {model_name}")
		
		if component_type == "CPU":
			df = self.cpu_data
			
			# Extract CPU details for better matching
			if "Intel" in model_name or "Core" in model_name:
				# Handle Intel CPUs 
				model_parts = model_name.replace("Intel ", "").strip()
				if "Core" not in model_parts and "i" in model_parts:
					model_parts = f"Core {model_parts}"
				
				# Extract just the model number if possible 
				model_number = None
				if "-" in model_parts:
					model_number = model_parts.split("-")[-1]
				
				search_terms = [
					model_name,
					model_parts,
					f"Core {model_parts}" if "Core" not in model_parts else model_parts,
					model_number if model_number else ""
				]
				
			elif "AMD" in model_name or "Ryzen" in model_name:
				# Handle AMD CPUs 
				model_parts = model_name.replace("AMD ", "").strip()
				
				# Extract just the model number if possible 
				model_number = None
				if len(model_parts.split()) >= 3:  # Format like "Ryzen 7 5800X"
					model_number = model_parts.split()[-1]
				
				search_terms = [
					model_name,
					model_parts,
					f"Ryzen {model_parts}" if "Ryzen" not in model_parts else model_parts,
					model_number if model_number else ""
				]
			else:
				# Generic case if we can't determine the brand
				search_terms = [model_name]
				
			# Remove empty search terms
			search_terms = [term for term in search_terms if term]
			
		elif component_type == "GPU":
			df = self.gpu_data
			search_terms = [
				model_name,
				f"NVIDIA {model_name}",
				f"AMD {model_name}",
				model_name.replace("NVIDIA ", "").replace("AMD ", "")
			]
		else:  # RAM
			df = self.ram_data
			# Extract RAM type and speed from model_name
			ram_type = None
			ram_speed = None
			
			try:
				if "DDR" in model_name:
					ram_parts = model_name.split()
					ram_type = ram_parts[0]  # e.g., "DDR5"
					if len(ram_parts) > 1:
						ram_speed = ram_parts[1].replace("MHz", "")  # e.g., "8000" from "8000MHz"
					
					search_terms = [
						model_name,
						f"{ram_type} {ram_speed}" if ram_speed else ram_type,  # "DDR5 8000"
						ram_speed if ram_speed else "",  # Just the speed number
						ram_type  # Just the type
					]
				else:
					search_terms = [model_name]
					
				# Remove empty search terms
				search_terms = [term for term in search_terms if term]
			except Exception as e:
				logger.error(f"Error parsing RAM model name: {str(e)}")
				search_terms = [model_name]
		
		logger.info(f"Search terms for {component_type}: {search_terms}")

		# Try each search term
		for term in search_terms:
			try:
				mask = df['Model'].str.contains(term, case=False, na=False)
				match_count = mask.sum()
				logger.info(f"Search term '{term}' found {match_count} matches")
				
				if match_count > 0:
					benchmark = float(df[mask].iloc[0]['Benchmark'])
					logger.info(f"Found benchmark for {component_type} {model_name}: {benchmark}")
					return benchmark
			except Exception as e:
				logger.error(f"Error searching for '{term}': {str(e)}")
		
		# If no match found by model name, try another approach for CPUs
		if component_type == "CPU" and not model_number:
			try:
				# Try to extract model number from the name
				import re
				for term in search_terms:
					# Find any numeric sequence followed by letters (like "13600K")
					matches = re.findall(r'\d+[A-Za-z]+', term)
					logger.info(f"Regex matches for '{term}': {matches}")
					if matches:
						for match in matches:
							mask = df['Model'].str.contains(match, case=False, na=False)
							match_count = mask.sum()
							logger.info(f"Regex match '{match}' found {match_count} matches")
							if match_count > 0:
								benchmark = float(df[mask].iloc[0]['Benchmark'])
								logger.info(f"Found benchmark via regex for {component_type} {model_name}: {benchmark}")
								return benchmark
			except Exception as e:
				logger.error(f"Error in regex search: {str(e)}")
		
		# If still no match found and it's RAM, use a fallback approach
		if component_type == "RAM" and ram_type:
			logger.info(f"Trying RAM fallback search for type: {ram_type}")
			try:
				# Try to match just by RAM type (DDR4, DDR5, etc)
				type_mask = df['Model'].str.contains(ram_type, case=False, na=False)
				type_matches = type_mask.sum()
				
				if type_matches > 0:
					# If RAM speed was provided, try to find the closest speed in that type
					if ram_speed:
						ram_speed_int = int(ram_speed)
						matches_df = df[type_mask].copy()
						
						# Try to extract speeds from model names
						import re
						
						def extract_speed(model_name):
							numbers = re.findall(r'\d+', model_name)
							if numbers:
								for num in numbers:
									if len(num) >= 4:  # Usually RAM speeds are 4+ digits
										return int(num)
								return int(numbers[0])  # Fallback to first number
							return 0
						
						# Try to find models with speed info and get the closest one
						if not matches_df.empty:
							matches_df['extracted_speed'] = matches_df['Model'].apply(extract_speed)
							
							# Filter out zeros
							matches_with_speed = matches_df[matches_df['extracted_speed'] > 0]
							
							if not matches_with_speed.empty:
								# Find closest speed
								matches_with_speed['speed_diff'] = abs(matches_with_speed['extracted_speed'] - ram_speed_int)
								closest_match = matches_with_speed.loc[matches_with_speed['speed_diff'].idxmin()]
								
								benchmark = float(closest_match['Benchmark'])
								logger.info(f"Found closest RAM benchmark by speed: {closest_match['Model']} with benchmark: {benchmark}")
								return benchmark
					
					# If we couldn't match by speed or speed wasn't provided, use the best rated RAM of that type
					best_match = df[type_mask].sort_values('Benchmark', ascending=False).iloc[0]
					benchmark = float(best_match['Benchmark'])
					logger.info(f"Using best RAM benchmark for type {ram_type}: {best_match['Model']} with benchmark: {benchmark}")
					return benchmark
			except Exception as e:
				logger.error(f"Error in RAM fallback search: {str(e)}")
		
		logger.warning(f"No benchmark found for {component_type} {model_name}")
		return None

	def validate_prediction(self, components):
		"""Validate prediction based on benchmark scores."""
		logger.info(f"Validating components: {components}")
		benchmarks = {}
		
		# Get benchmark scores
		for component, model in components.items():
			score = self.find_benchmark(component, model)
			if score is not None:
				benchmarks[component] = score
			else:
				logger.warning(f"Missing benchmark for {component}, validation cannot proceed")
				return None  # Cannot validate if missing benchmarks

		logger.info(f"Collected benchmarks: {benchmarks}")
		
		# Calculate relative performance differences
		try:
			max_score = max(benchmarks.values())
			relative_scores = {
				comp: score / max_score * 100 
				for comp, score in benchmarks.items()
			}

			# Determine bottleneck based on lowest relative score
			bottleneck = min(relative_scores.items(), key=lambda x: x[1])
			
			logger.info(f"Relative scores: {relative_scores}")
			logger.info(f"Identified hardware bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f}%)")
			
			return {
				'hardware_bottleneck': bottleneck[0],
				'relative_scores': relative_scores
			}
		except Exception as e:
			logger.error(f"Error calculating validation result: {str(e)}")
			return None

