import re
import pandas as pd
import ast  # Per convertire la stringa del dizionario in un vero dizionario

def parse_optuna_log(log_file_path):
    # Regex per catturare: Trial ID, Value (Score), e Parameters
    # Cerca righe come: "Trial 10 finished with value: 0.216... and parameters: {'topK': 88, ...}."
    regex_pattern = r"Trial (\d+) finished with value: ([\d\.]+) and parameters: (\{.*?\})\."
    
    data = []
    processed_trials = set()

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(regex_pattern, line)
            if match:
                trial_id = int(match.group(1))

                if trial_id in processed_trials:
                    continue
                
                processed_trials.add(trial_id)
                
                value = float(match.group(2))
                params_str = match.group(3)
                
                # Converte la stringa dei parametri in un dizionario Python sicuro
                try:
                    params = ast.literal_eval(params_str)
                except:
                    params = {}

                # Uniamo tutto in un unico record
                record = {
                    'Trial_ID': trial_id,
                    'Value': value,
                    **params  # Espande il dizionario parametri in colonne separate
                }
                data.append(record)

    # Creazione DataFrame
    df = pd.DataFrame(data)
    return df

# --- Utilizzo ---
# Sostituisci con il percorso del tuo file
file_path = 'itemknn-par-opt-3.log' 

try:
    df_results = parse_optuna_log(file_path)
    
    # Ordina per risultato migliore (decrescente)
    df_results = df_results.sort_values(by='Value', ascending=False)
    
    print("Top 5 Risultati:")
    print(df_results.head(5))
    
    # Salva in CSV
    df_results.to_csv('optuna_results.csv', index=False)
    print("\nFile 'optuna_results.csv' salvato con successo.")
    
except FileNotFoundError:
    print("Errore: File non trovato. Controlla il percorso.")

file_path = 'itemknn-par-opt-3.log'

if __name__ == "__main__":
    df_results = parse_optuna_log(file_path)
    df_results = df_results.sort_values(by='Value', ascending=False)
    print(df_results.head(60))

import matplotlib.pyplot as plt

# 1. Filtra solo i risultati TF-IDF (i più rilevanti)
#df_plot = df_results[df_results['feature_weighting'] == 'TF-IDF'].copy()

"""# 2. Crea il grafico
plt.figure(figsize=(10, 8))

# Scatter plot: x=topK, y=shrink, c=Value (colore)
sc = plt.scatter(df_plot['topK'], df_plot['shrink'], 
                 c=df_plot['Value'], cmap='viridis', 
                 s=60, alpha=0.9, edgecolors='k', linewidth=0.5)

# Evidenzia il punto migliore con una stella rossa
best_row = df_plot.loc[df_plot['Value'].idxmax()]
plt.scatter(best_row['topK'], best_row['shrink'], color='red', s=200, marker='*', 
            label=f"BEST Trial {best_row['Trial_ID']}: {best_row['Value']:.5f}\n(topK={best_row['topK']}, shrink={best_row['shrink']})")

# Aggiungi colorbar e etichette
plt.colorbar(sc, label='Score')
plt.xlabel('topK')
plt.ylabel('shrink')
plt.title('Hyperparameter Optimization Landscape (TF-IDF Only)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right')

plt.show()"""