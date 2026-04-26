#!/usr/bin/env python3
"""
Field Assistant - Natural language interface for surveyor queries.

Provides rule-based NLP (no LLM required) for:
  - Grade estimation queries
  - Volume calculation requests
  - Anomaly detection queries
  - Coordinate transformation
  - Offline-capable
"""

import re
import numpy as np


class FieldAssistant:
    """Natural language interface for field operations."""
    
    def __init__(self, grade_estimator=None, volume_calculator=None, anomaly_detector=None):
        """
        Initialize field assistant with analysis tools.
        
        Args:
            grade_estimator: GradeEstimator instance
            volume_calculator: VolumeCalculator instance
            anomaly_detector: AnomalyDetector instance
        """
        self.grade_estimator = grade_estimator
        self.volume_calculator = volume_calculator
        self.anomaly_detector = anomaly_detector
        self.command_history = []
    
    def parse_query(self, query):
        """
        Parse natural language query into structured command.
        
        Returns:
            {
                'action': 'estimate_grade|calculate_volume|detect_anomaly|...',
                'parameters': {...},
                'natural_language': query
            }
        """
        query_lower = query.lower().strip()
        
        # Grade estimation queries
        if any(word in query_lower for word in ['grade', 'estimate', 'assay', 'cu']):
            # Pattern: "estimate grade at X, Y, Z"
            coords_match = re.search(r'at\s+([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)', query_lower)
            if coords_match:
                return {
                    'action': 'estimate_grade',
                    'parameters': {
                        'x': float(coords_match.group(1)),
                        'y': float(coords_match.group(2)),
                        'z': float(coords_match.group(3))
                    }
                }
        
        # Volume calculation queries
        elif any(word in query_lower for word in ['volume', 'cut', 'fill', 'earthwork']):
            # Pattern: "calculate volume between (x1,y1,z1) and (x2,y2,z2)"
            coords_match = re.findall(r'([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)', query_lower)
            if len(coords_match) >= 2:
                return {
                    'action': 'calculate_volume',
                    'parameters': {
                        'point1': tuple(map(float, coords_match[0])),
                        'point2': tuple(map(float, coords_match[1]))
                    }
                }
        
        # Anomaly detection queries
        elif any(word in query_lower for word in ['anomaly', 'outlier', 'unusual', 'error', 'check']):
            if 'grade' in query_lower or 'assay' in query_lower:
                return {
                    'action': 'detect_grade_anomalies',
                    'parameters': {}
                }
            elif 'coordin' in query_lower or 'location' in query_lower:
                return {
                    'action': 'detect_coordinate_anomalies',
                    'parameters': {}
                }
            else:
                return {
                    'action': 'detect_anomalies',
                    'parameters': {}
                }
        
        # Coordinate transformation queries
        elif any(word in query_lower for word in ['convert', 'transform', 'coordinate', 'utm', 'lat', 'lon']):
            if 'geographic' in query_lower or 'lat' in query_lower or 'lon' in query_lower:
                coords_match = re.findall(r'([\d.-]+)', query_lower)
                if len(coords_match) >= 2:
                    return {
                        'action': 'convert_to_utm',
                        'parameters': {
                            'latitude': float(coords_match[0]),
                            'longitude': float(coords_match[1])
                        }
                    }
        
        # Help request
        elif any(word in query_lower for word in ['help', '?', 'what can', 'how to']):
            return {
                'action': 'help',
                'parameters': {}
            }
        
        # Default: echo back with uncertainty
        return {
            'action': 'unknown',
            'parameters': {},
            'confidence': 0.0
        }
    
    def execute_command(self, command):
        """
        Execute parsed command.
        
        Args:
            command: dict from parse_query()
        
        Returns:
            response string
        """
        action = command['action']
        params = command['parameters']
        
        try:
            if action == 'estimate_grade':
                if self.grade_estimator is None:
                    return "Grade estimator not initialized. Please train model first."
                
                x, y, z = params['x'], params['y'], params['z']
                X = np.array([[x, y, z]])
                pred, ci_low, ci_high = self.grade_estimator.confidence_interval(X)
                
                return f"Estimated grade at ({x}, {y}, {z}): {pred[0]:.2f} ppm Cu\nConfidence interval (95%): {ci_low[0]:.2f} - {ci_high[0]:.2f} ppm Cu"
            
            elif action == 'calculate_volume':
                if self.volume_calculator is None:
                    return "Volume calculator not initialized."
                
                p1, p2 = params['point1'], params['point2']
                return f"Volume calculation requested between {p1} and {p2}\nUse calculate_cut_fill() method for detailed results."
            
            elif action == 'detect_grade_anomalies':
                if self.anomaly_detector is None:
                    return "Anomaly detector not initialized."
                
                return f"Grade anomalies: {len(self.anomaly_detector.anomalies.get('grade', {}).get('indices', []))} detected\nCheck report for details."
            
            elif action == 'detect_coordinate_anomalies':
                if self.anomaly_detector is None:
                    return "Anomaly detector not initialized."
                
                return f"Coordinate anomalies: {len(self.anomaly_detector.anomalies.get('coordinate', {}).get('indices', []))} detected\nCheck report for details."
            
            elif action == 'help':
                return self._get_help_text()
            
            else:
                return "I didn't understand that command. Try 'help' for available commands."
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_query(self, query):
        """
        Process natural language query end-to-end.
        
        Args:
            query: natural language string
        
        Returns:
            response string
        """
        self.command_history.append(query)
        command = self.parse_query(query)
        response = self.execute_command(command)
        return response
    
    def _get_help_text(self):
        """Get help text."""
        return """
ASST SURVEYOR AI - FIELD ASSISTANT
===================================

Available Commands:

1. GRADE ESTIMATION
   "Estimate grade at 800, 600, 150"
   "What's the copper grade at point X,Y,Z?"
   
2. VOLUME CALCULATION
   "Calculate volume between (800,600,150) and (900,700,200)"
   "Cut and fill between surface A and B"
   
3. ANOMALY DETECTION
   "Check for grade anomalies"
   "Are there coordinate errors in this dataset?"
   "Detect outliers in the assay data"
   
4. COORDINATE CONVERSION
   "Convert 6.2, 106.8 to UTM"
   "What's the UTM coordinate for this lat/lon?"
   
5. HELP
   Type "help" or "?" to see this message

TIPS:
- Provide coordinates as: X, Y, Z or X,Y,Z
- Separate multiple coordinates with "and"
- Offline operation - no internet required
"""
    
    def interactive_session(self):
        """Start interactive query session."""
        print("\n" + "="*60)
        print("ASST SURVEYOR AI - FIELD ASSISTANT (Interactive Mode)")
        print("="*60)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                query = input("Query> ").strip()
                
                if query.lower() == 'quit':
                    print("Exiting field assistant.")
                    break
                
                if not query:
                    continue
                
                response = self.process_query(query)
                print(f"\nResponse: {response}\n")
            
            except KeyboardInterrupt:
                print("\nExiting field assistant.")
                break
            except Exception as e:
                print(f"Error: {e}\n")
