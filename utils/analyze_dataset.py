import os
from collections import Counter

def main():
    data_dir = r"dataset/Melops_head/Melops_head"
    metadata_path = "Melops_metadata.txt"
    output_file = "statistiche_pesci.txt"
    
    if not os.path.exists(data_dir):
        print(f"Cartella non trovata: {data_dir}")
        return

    print("Caricamento metadati...")
    filename_to_id = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            next(f)  # Salta intestazione
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    fname = parts[0].strip('"')
                    true_id = parts[4].strip('"')
                    filename_to_id[fname] = true_id
    else:
        print(f"Attenzione: file {metadata_path} non trovato!")

    print("Analisi del dataset in corso usando i veri ID...")
    
    # Raccogliamo tutti i veri ID
    ids = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            base_name = os.path.splitext(filename)[0] # es. "1160507_2018"
            
            # Recuperiamo il vero ID dai metadati
            true_id = filename_to_id.get(base_name, base_name)
            ids.append(true_id)
            
    # Contiamo quante immagini ci sono per ogni ID
    id_counts = Counter(ids)
    
    # Filtriamo solo i pesci che hanno più di 1 foto (quindi possono formare coppie positive)
    multiple_images = {k: v for k, v in id_counts.items() if v > 1}
    
    # Ordiniamo i pesci per numero di foto decrescente
    sorted_multiple = sorted(multiple_images.items(), key=lambda item: item[1], reverse=True)
    
    total_images = len(ids)
    total_unique_fish = len(id_counts)
    fish_with_pairs = len(multiple_images)
    
    # Calcolo totale combinazioni possibili di coppie positive
    # Formula: N * (N - 1) / 2
    total_positive_pairs = sum((count * (count - 1)) // 2 for count in multiple_images.values())
    
    # Scriviamo i risultati su un file di testo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== STATISTICHE DATASET MELOPS ===\n")
        f.write(f"Totale immagini trovate: {total_images}\n")
        f.write(f"Totale pesci (ID univoci): {total_unique_fish}\n")
        f.write(f"Pesci con più di 1 foto (utili per coppie positive): {fish_with_pairs}\n")
        f.write(f"Numero totale di coppie positive uniche possibili: {total_positive_pairs}\n")
        f.write("-" * 40 + "\n")
        f.write("Dettaglio pesci con più di una foto:\n")
        f.write("ID PESCE\t->\tNUMERO FOTO\n")
        
        for fish_id, count in sorted_multiple:
            f.write(f"{fish_id}\t\t->\t{count} foto\n")
            
    print(f"Analisi completata! I risultati sono stati salvati nel file '{output_file}'.")

if __name__ == "__main__":
    main()
