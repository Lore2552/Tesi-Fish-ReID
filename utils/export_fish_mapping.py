import os

def main():
    data_dir = r"dataset/Melops_head/Melops_head"
    metadata_path = "Melops_metadata.txt"
    output_file = "mappatura_pesci.txt"
    
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

    print("Raggruppamento delle foto per ID...")
    
    # Raccogliamo i file per ID
    images_by_id = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            base_name = os.path.splitext(filename)[0]
            
            # Recuperiamo il vero ID dai metadati
            true_id = filename_to_id.get(base_name, base_name)
            
            if true_id not in images_by_id:
                images_by_id[true_id] = []
            images_by_id[true_id].append(filename)
            
    # Filtriamo tenendo solo i pesci che compaiono in PIÙ DI UNA foto
    multiple_images = {k: v for k, v in images_by_id.items() if len(v) > 1}
    
    # Ordiniamo in base a chi ha più foto
    sorted_multiple = sorted(multiple_images.items(), key=lambda item: len(item[1]), reverse=True)
    
    print("Scrittura del file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== MAPPATURA ID -> FOTO ===\n")
        f.write("Questo file mostra, per ogni esemplare di pesce (ID univoco), l'elenco esatto delle foto in cui compare.\n")
        f.write("Sono inclusi solo i pesci con foto multiple.\n\n")
        
        for fish_id, filenames in sorted_multiple:
            # Uniamo i nomi dei file con una virgola
            files_str = ", ".join(filenames)
            f.write(f"ID PESCE {fish_id} ({len(filenames)} foto): {files_str}\n")
            
    print(f"Completato! La mappatura è stata salvata in: {output_file}")

if __name__ == "__main__":
    main()
