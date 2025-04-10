"""
Gardening Company Example

This script demonstrates how the Iris classification model could be used
in a gardening company for plant classification and care recommendations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import mlflow
import mlflow.sklearn

# Import functions from iris_model.py
from iris_model import load_and_evaluate_model, FEATURE_NAMES, CLASS_NAMES, EXPERIMENT_NAME

class GardeningCompanyWorkflow:
    """
    Simulate a gardening company's workflow that uses the iris model
    for plant identification and classification
    """
    
    def __init__(self):
        """Initialize the workflow"""
        # Set MLflow tracking URI to point to the MLflow server
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        
        # Load the model
        self.model, _, _ = load_and_evaluate_model()
        
        # Initialize database files
        self.plants_database = "plants_database.csv"
        self.measurements_log = "measurements_log.csv"
        self.care_recommendations_log = "care_recommendations.csv"
        
        # Initialize database files if they don't exist
        self._initialize_databases()
    
    def _initialize_databases(self):
        """Create dummy database files for demonstration"""
        # Create plants database if it doesn't exist
        if not os.path.exists(self.plants_database):
            plants = []
            for _ in range(20):  # Create 20 sample plants
                plant_type = random.choice(CLASS_NAMES)
                
                # Generate realistic measurements based on the iris class
                if plant_type == "setosa":
                    measurements = [
                        round(random.uniform(4.5, 5.5), 1),  # Sepal length
                        round(random.uniform(3.0, 4.0), 1),  # Sepal width
                        round(random.uniform(1.0, 1.7), 1),  # Petal length
                        round(random.uniform(0.1, 0.3), 1)   # Petal width
                    ]
                elif plant_type == "versicolor":
                    measurements = [
                        round(random.uniform(5.5, 6.5), 1),  # Sepal length
                        round(random.uniform(2.5, 3.0), 1),  # Sepal width
                        round(random.uniform(3.5, 4.5), 1),  # Petal length
                        round(random.uniform(1.0, 1.5), 1)   # Petal width
                    ]
                else:  # virginica
                    measurements = [
                        round(random.uniform(6.5, 7.5), 1),  # Sepal length
                        round(random.uniform(2.8, 3.2), 1),  # Sepal width
                        round(random.uniform(5.0, 6.0), 1),  # Petal length
                        round(random.uniform(1.5, 2.0), 1)   # Petal width
                    ]
                
                plant_id = f"PLANT-{random.randint(1000, 9999)}"
                plant_data = {
                    "plant_id": plant_id,
                    "plant_type": plant_type,
                    "planting_date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
                    "location": random.choice(["Garden A", "Garden B", "Greenhouse 1", "Greenhouse 2"]),
                    "last_measured": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
                    "sepal_length_cm": measurements[0],
                    "sepal_width_cm": measurements[1],
                    "petal_length_cm": measurements[2],
                    "petal_width_cm": measurements[3],
                    "health_status": random.choice(["Excellent", "Good", "Fair", "Poor"]),
                    "notes": ""
                }
                plants.append(plant_data)
            
            # Save to CSV
            plants_df = pd.DataFrame(plants)
            plants_df.to_csv(self.plants_database, index=False)
            print(f"Created plants database with {len(plants)} plants")
        
        # Create measurement log if it doesn't exist
        if not os.path.exists(self.measurements_log):
            with open(self.measurements_log, 'w') as f:
                f.write("timestamp,plant_id,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,predicted_type,confidence\n")
            print("Created empty measurements log")
        
        # Create care recommendations log if it doesn't exist
        if not os.path.exists(self.care_recommendations_log):
            with open(self.care_recommendations_log, 'w') as f:
                f.write("timestamp,plant_id,plant_type,recommended_action,notes\n")
            print("Created empty care recommendations log")
    
    def collect_new_measurements(self):
        """
        Simulate the collection of new plant measurements.
        In a real scenario, this would come from IoT sensors or manual measurements.
        """
        print("\n=== Collecting New Measurements ===")
        plants_df = pd.read_csv(self.plants_database)
        
        # Select a random subset of plants to measure today
        num_plants_to_measure = random.randint(3, 8)
        plants_to_measure = plants_df.sample(min(num_plants_to_measure, len(plants_df)))
        
        # Generate new measurements with some random variation
        new_measurements = []
        for _, plant in plants_to_measure.iterrows():
            # Get base measurements
            measurements = [
                plant["sepal_length_cm"],
                plant["sepal_width_cm"],
                plant["petal_length_cm"],
                plant["petal_width_cm"]
            ]
            
            # Add small random variations (simulate plant growth and measurement error)
            measurements = [
                round(m + random.uniform(-0.2, 0.3), 1) for m in measurements
            ]
            
            # Make sure values stay positive
            measurements = [max(0.1, m) for m in measurements]
            
            new_measurements.append({
                "plant_id": plant["plant_id"],
                "sepal_length_cm": measurements[0],
                "sepal_width_cm": measurements[1],
                "petal_length_cm": measurements[2],
                "petal_width_cm": measurements[3]
            })
        
        print(f"Collected measurements for {len(new_measurements)} plants")
        return new_measurements, plants_df
    
    def process_measurements(self, new_measurements, plants_df):
        """Process new measurements and update the database"""
        print("\n=== Processing Measurements ===")
        
        # Convert to DataFrame for prediction
        measurements_df = pd.DataFrame(new_measurements)
        features_df = measurements_df[FEATURE_NAMES]
        
        # Make predictions using our model
        predictions = self.model.predict(features_df)
        probabilities = self.model.predict_proba(features_df)
        
        # Combine measurements with predictions
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log measurements and predictions
        with open(self.measurements_log, 'a') as f:
            for i, measurement in enumerate(new_measurements):
                confidence = probabilities[i][predictions[i]]
                predicted_class = CLASS_NAMES[predictions[i]]
                # Convert all values to strings explicitly to avoid pandas casting warnings
                plant_id = str(measurement['plant_id'])
                sepal_length = str(measurement['sepal_length_cm'])
                sepal_width = str(measurement['sepal_width_cm'])
                petal_length = str(measurement['petal_length_cm'])
                petal_width = str(measurement['petal_width_cm'])
                confidence_str = f"{confidence:.4f}"
                f.write(f"{timestamp},{plant_id},{sepal_length},{sepal_width},{petal_length},{petal_width},{predicted_class},{confidence_str}\n")
        
        # Update plant database with new measurements
        for i, measurement in enumerate(new_measurements):
            plant_id = measurement['plant_id']
            predicted_class = CLASS_NAMES[predictions[i]]
            confidence = probabilities[i][predictions[i]]
            
            # Find the plant in the database
            idx = plants_df.index[plants_df['plant_id'] == plant_id].tolist()
            if idx:
                # Update measurements
                plants_df.loc[idx[0], 'sepal_length_cm'] = measurement['sepal_length_cm']
                plants_df.loc[idx[0], 'sepal_width_cm'] = measurement['sepal_width_cm']
                plants_df.loc[idx[0], 'petal_length_cm'] = measurement['petal_length_cm']
                plants_df.loc[idx[0], 'petal_width_cm'] = measurement['petal_width_cm']
                plants_df.loc[idx[0], 'last_measured'] = timestamp.split()[0]  # Just the date
                
                # Update plant type if confidence is high enough
                if confidence > 0.8:
                    if predicted_class != plants_df.loc[idx[0], 'plant_type']:
                        plants_df.loc[idx[0], 'plant_type'] = predicted_class
                        plants_df.loc[idx[0], 'notes'] = f"Plant type updated to {predicted_class} based on measurements on {timestamp}"
                        print(f"Plant {plant_id} reclassified as {predicted_class} with confidence {confidence:.4f}")
                    else:
                        print(f"Plant {plant_id} confirmed as {predicted_class} with confidence {confidence:.4f}")
                else:
                    print(f"Plant {plant_id} measured but classification uncertain (confidence: {confidence:.4f})")
        
        # Save updated plants database
        plants_df.to_csv(self.plants_database, index=False)
        
        return plants_df
    
    def generate_care_recommendations(self, plants_df):
        """
        Generate plant care recommendations based on the latest measurements
        and classifications
        """
        print("\n=== Generating Care Recommendations ===")
        
        care_recommendations = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for _, plant in plants_df.iterrows():
            plant_type = plant["plant_type"]
            health_status = plant["health_status"]
            
            # Generate care recommendations based on plant type
            if plant_type == "setosa":
                if health_status in ["Poor", "Fair"]:
                    action = random.choice([
                        "Increase watering frequency to twice per week",
                        "Apply specific setosa fertilizer",
                        "Check for pests common to setosa varieties"
                    ])
                else:
                    action = random.choice([
                        "Maintain standard setosa care protocol",
                        "Monitor growth patterns",
                        "Consider propagating healthy specimens"
                    ])
            elif plant_type == "versicolor":
                if health_status in ["Poor", "Fair"]:
                    action = random.choice([
                        "Increase humidity levels around the plant",
                        "Apply balanced NPK fertilizer",
                        "Check soil pH, ensure 6.0-6.5 range"
                    ])
                else:
                    action = random.choice([
                        "Maintain current care routine",
                        "Consider trimming after flowering season",
                        "Protect from excessive sunlight"
                    ])
            else:  # virginica
                if health_status in ["Poor", "Fair"]:
                    action = random.choice([
                        "Check for root health and repot if necessary",
                        "Apply specialized virginica nutrients",
                        "Increase light exposure gradually"
                    ])
                else:
                    action = random.choice([
                        "Maintain ideal moisture levels",
                        "Apply preventative pest measures",
                        "Consider dividing larger specimens"
                    ])
            
            # Only generate recommendations for plants that have been measured in the last 30 days
            last_measured = datetime.strptime(plant["last_measured"], "%Y-%m-%d")
            if (datetime.now() - last_measured).days <= 30:
                care_recommendations.append({
                    "timestamp": timestamp,
                    "plant_id": plant["plant_id"],
                    "plant_type": plant_type,
                    "recommended_action": action,
                    "notes": f"Based on {health_status} health status"
                })
        
        # Save recommendations to log
        with open(self.care_recommendations_log, 'a') as f:
            for rec in care_recommendations:
                f.write(f"{rec['timestamp']},{rec['plant_id']},{rec['plant_type']},{rec['recommended_action']},{rec['notes']}\n")
        
        print(f"Generated {len(care_recommendations)} plant care recommendations")
        
        # Print a few examples
        if care_recommendations:
            print("\nExample Care Recommendations:")
            for i, rec in enumerate(care_recommendations[:3]):  # Show up to 3 examples
                print(f"{i+1}. Plant {rec['plant_id']} ({rec['plant_type']}): {rec['recommended_action']}")
        
        return care_recommendations
    
    def monitor_model_performance(self):
        """
        Monitor model performance and suggest retraining if necessary.
        In a real scenario, this would compare predictions against actual observations.
        """
        print("\n=== Monitoring Model Performance ===")
        
        # Read recent measurements
        try:
            measurements_df = pd.read_csv(self.measurements_log)
            
            # Get the most recent 20 measurements or all if less than 20
            recent_measurements = measurements_df.tail(20)
            
            # Extract confidence values
            if 'confidence' in recent_measurements.columns:
                confidences = recent_measurements['confidence'].astype(float)
                avg_confidence = confidences.mean()
                high_confidence = confidences[confidences > 0.8]
                confidence_ratio = len(high_confidence) / len(confidences)
                
                print(f"Average prediction confidence: {avg_confidence:.4f}")
                print(f"Percentage of high-confidence predictions: {confidence_ratio:.2%}")
                
                # If less than 70% of predictions have high confidence, suggest retraining
                if confidence_ratio < 0.7:
                    print("\nWARNING: Model confidence is low. Consider retraining the model with recent data.")
                    return False
                else:
                    print("\nModel performance is satisfactory.")
                    return True
            else:
                print("No confidence data available.")
                return True
        except Exception as e:
            print(f"Error monitoring model performance: {str(e)}")
            return False
    
    def run_daily_workflow(self):
        """Run the daily workflow"""
        print(f"\n=== Running daily workflow at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # Step 1: Collect new measurements
        new_measurements, plants_df = self.collect_new_measurements()
        
        # Step 2: Process measurements and update database
        plants_df = self.process_measurements(new_measurements, plants_df)
        
        # Step 3: Generate care recommendations
        care_recommendations = self.generate_care_recommendations(plants_df)
        
        # Step 4: Monitor model performance
        model_performing_well = self.monitor_model_performance()
        
        print("\n=== Daily workflow completed ===")
        print(f"Processed {len(new_measurements)} plant measurements")
        print(f"Generated {len(care_recommendations)} care recommendations")
        print(f"Model performance is {'good' if model_performing_well else 'needs attention'}")
        
        return model_performing_well

def simulate_workflow(days=3):
    """Simulate a few days of operation"""
    print("\n=== Gardening Company Workflow Simulation ===")
    print("This simulation demonstrates how the Iris classification model")
    print("could be used in a gardening company's daily operations.")
    print(f"Simulating {days} days of garden operations...")
    
    workflow = GardeningCompanyWorkflow()
    
    for day in range(1, days + 1):
        print(f"\n\n{'='*20} Day {day} {'='*20}")
        workflow.run_daily_workflow()
    
    print("\n=== Workflow simulation completed ===")
    print("In a real implementation, this would run continuously as a scheduled task,")
    print("processing new measurements daily and providing care recommendations.")
    print("The model would be monitored for performance and retrained as needed.")

def explain_workflow_benefits():
    """Explain the benefits of this workflow for a gardening company"""
    print("\n=== Benefits of MLflow-based Workflow for a Gardening Company ===")
    print("""
1. Automated Plant Classification:
   - Accurately identify plant species based on measurements
   - Reduce human error in classification
   - Process large numbers of plants efficiently

2. Personalized Care Recommendations:
   - Generate species-specific care instructions
   - Adapt recommendations based on plant health
   - Ensure consistent care protocols across staff

3. Continuous Improvement:
   - Track model performance over time
   - Automatically detect when retraining is needed
   - Incorporate new data to improve accuracy

4. Operational Benefits:
   - Maintain detailed records of all plants and measurements
   - Track plant health trends over time
   - Optimize resource allocation (water, fertilizer, staff time)

5. MLflow Integration Benefits:
   - Version control for models
   - Track experiments and improvements
   - Easy model deployment and updates
   - Comprehensive performance metrics
    """)

if __name__ == "__main__":
    # Simulate the workflow
    simulate_workflow(days=3)
    
    # Explain the benefits
    explain_workflow_benefits()
