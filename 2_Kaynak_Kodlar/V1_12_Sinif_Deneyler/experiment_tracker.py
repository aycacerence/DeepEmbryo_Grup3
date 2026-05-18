import os
import pandas as pd

def log_experiment(exp_name, history, test_results):
    """
    Logs experiment results to a CSV file.
    """
    results_file = "experiments_results.csv"
    
    # Extract best metrics
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
    
    new_data = {
        'Experiment': [exp_name],
        'Best_Epoch': [best_epoch + 1],
        'Train_Acc': [history.history['accuracy'][best_epoch]],
        'Val_Acc': [history.history['val_accuracy'][best_epoch]],
        'Test_Acc': [test_results[1]],
        'Test_Loss': [test_results[0]]
    }
    
    df_new = pd.DataFrame(new_data)
    
    if os.path.exists(results_file):
        df_old = pd.read_csv(results_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
        
    df_final.to_csv(results_file, index=False)
    print(f"Results for {exp_name} logged to {results_file}")

if __name__ == "__main__":
    # Test logging
    print("Experiment Tracker Module Ready.")
