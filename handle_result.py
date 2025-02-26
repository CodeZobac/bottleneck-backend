import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_percentile_rank(component_type, benchmark_score, benchmark_csv):
    """Calculate the percentile rank of a component within its category."""
    try:
        df = pd.read_csv(benchmark_csv)
        # Higher benchmark is better, so we want to know what percentage of components this one beats
        percentile = (df[df['Benchmark'] <= benchmark_score].shape[0] / df.shape[0]) * 100
        return 100 - percentile  # Convert to "top X%" format (lower is better)
    except Exception as e:
        logger.error(f"Error calculating percentile for {component_type}: {str(e)}")
        return None

def interpret_hardware_scores(hardware_scores, cpu_csv, gpu_csv, ram_csv):
    """Interpret hardware scores to determine the true bottleneck."""
    percentile_ranks = {}
    
    if 'CPU' in hardware_scores:
        percentile_ranks['CPU'] = get_percentile_rank('CPU', hardware_scores['CPU'], cpu_csv)
    
    if 'GPU' in hardware_scores:
        percentile_ranks['GPU'] = get_percentile_rank('GPU', hardware_scores['GPU'], gpu_csv)
    
    if 'RAM' in hardware_scores:
        percentile_ranks['RAM'] = get_percentile_rank('RAM', hardware_scores['RAM'], ram_csv)
    
    logger.info(f"Component percentile ranks: {percentile_ranks}")
    
    # Identify the weakest component (highest percentile number = furthest from top)
    if percentile_ranks:
        bottleneck = max(percentile_ranks.items(), key=lambda x: x[1])
        logger.info(f"Adjusted bottleneck based on percentile ranks: {bottleneck[0]}")
        return {
            'hardware_bottleneck': bottleneck[0],
            'percentile_ranks': percentile_ranks
        }
    return None

def estimate_performance_impact(percentile_ranks):
    """Estimate the performance impact of each component."""
    if not percentile_ranks:
        return {}
    
    # Calculate the gap between each component and the best component
    best_percentile = min(percentile_ranks.values())
    impact = {
        component: (rank - best_percentile) / 2  # Scale down a bit for more reasonable impact numbers
        for component, rank in percentile_ranks.items()
    }
    
    # Ensure no negative impacts
    impact = {k: max(0, v) for k, v in impact.items()}
    
    return impact

def format_response(prediction_result):
    """Format the prediction result for API response."""
    response = {
        'ml_prediction': {
            'bottleneck': prediction_result.get('ml_bottleneck'),
            'confidence': prediction_result.get('ml_probabilities')
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # If we have hardware validation data
    if 'hardware_scores' in prediction_result:
        raw_scores = prediction_result['hardware_scores']
        
        # Get the adjusted interpretation
        interpretation = interpret_hardware_scores(
            raw_scores,
            './data/CPU_UserBenchmarks.csv',
            './data/GPU_UserBenchmarks.csv',
            './data/RAM_UserBenchmarks.csv'
        )
        
        if interpretation:
            hardware_bottleneck = interpretation['hardware_bottleneck']
            percentile_ranks = interpretation['percentile_ranks']
            
            # Estimate performance impact
            impact = estimate_performance_impact(percentile_ranks)
            
            response['hardware_analysis'] = {
                'bottleneck': hardware_bottleneck,
                'percentile_ranks': percentile_ranks,
                'raw_benchmark_scores': raw_scores,
                'estimated_impact': impact
            }
            
            # Check if ML and hardware analysis agree
            response['agreement'] = prediction_result.get('ml_bottleneck') == hardware_bottleneck
        else:
            response['hardware_analysis'] = {
                'bottleneck': prediction_result.get('hardware_bottleneck'),
                'raw_benchmark_scores': raw_scores
            }
            response['agreement'] = prediction_result.get('validation_match', False)
    
    # Add component quality assessment
    response['components'] = extract_component_quality(percentile_ranks if 'percentile_ranks' in locals() else None)
    
    return response

def extract_component_quality(percentile_ranks):
    """Extract quality assessment of each component."""
    if not percentile_ranks:
        return {}
    
    quality_levels = {
        'CPU': 'unknown',
        'GPU': 'unknown',
        'RAM': 'unknown'
    }
    
    for component, rank in percentile_ranks.items():
        if rank < 5:  # Top 5%
            quality_levels[component] = 'excellent'
        elif rank < 20:  # Top 20%
            quality_levels[component] = 'very good'
        elif rank < 50:  # Top 50%
            quality_levels[component] = 'good'
        elif rank < 80:  # Top 80%
            quality_levels[component] = 'average'
        else:
            quality_levels[component] = 'below average'
    
    return quality_levels

def process_prediction(prediction_result):
    """Process prediction result and format response for frontend."""
    logger.info("Processing prediction result")
    
    # Format the response
    response = format_response(prediction_result)
    
    # Add recommendations based on the bottleneck
    response['recommendations'] = generate_recommendations(response)
    
    return response

def generate_recommendations(response):
    """Generate recommendations based on the analysis."""
    recommendations = []
    
    # Determine which bottleneck to use (prefer hardware if available)
    bottleneck = None
    if 'hardware_analysis' in response and response['hardware_analysis'].get('bottleneck'):
        bottleneck = response['hardware_analysis']['bottleneck']
    elif 'ml_prediction' in response:
        bottleneck = response['ml_prediction']['bottleneck']
    
    if not bottleneck:
        return ["Insufficient data to generate recommendations."]
    
    # Generate component-specific recommendations
    if bottleneck == 'CPU':
        recommendations.append("Your CPU appears to be the limiting factor in your system.")
        if response.get('components', {}).get('CPU') in ['below average', 'average']:
            recommendations.append("Consider upgrading to a more powerful CPU to better balance your system.")
        else:
            recommendations.append("Despite having a relatively good CPU, it's still the bottleneck in your high-performance system.")
    
    elif bottleneck == 'GPU':
        recommendations.append("Your GPU appears to be the limiting factor in your system.")
        if response.get('components', {}).get('GPU') in ['below average', 'average']:
            recommendations.append("A GPU upgrade would likely yield the most significant performance improvement.")
        else:
            recommendations.append("Your system is well-balanced, but the GPU is slightly behind other components.")
    
    elif bottleneck == 'RAM':
        recommendations.append("Your RAM appears to be the limiting factor in your system.")
        if response.get('components', {}).get('RAM') in ['below average', 'average']:
            recommendations.append("Consider upgrading to faster RAM with higher frequency or adding more capacity.")
        else:
            recommendations.append("Your RAM is good, but still slightly limiting your system's potential.")
    
    # Add general advice
    recommendations.append("For optimal performance, aim to balance your system components.")
    
    return recommendations
