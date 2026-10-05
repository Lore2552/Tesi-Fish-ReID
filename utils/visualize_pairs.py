import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def main():
    data_dir = r"dataset/Melops_head/Melops_head"
    metadata_path = "Melops_metadata.txt"
    output_image = "verifica_identita.png"
    
    if not os.path.exists(data_dir):
        print("Cartella dataset non trovata.")
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

    # Raccogliamo le immagini per VERO ID
    images_by_id = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            base_name = os.path.splitext(filename)[0]
            true_id = filename_to_id.get(base_name, base_name)
            
            path = os.path.join(data_dir, filename)
            if true_id not in images_by_id:
                images_by_id[true_id] = []
            images_by_id[true_id].append(path)
            
    # Filtriamo solo gli ID che hanno tra le 2 e le 5 foto (per comodità di visualizzazione)
    valid_ids = [fish_id for fish_id, paths in images_by_id.items() if 1 < len(paths) <= 6]
    
    if not valid_ids:
        print("Nessun pesce con foto multiple trovato.")
        return
        
    # Scegliamo 3 pesci (ID) a caso per la visualizzazione
    selected_ids = random.sample(valid_ids, min(3, len(valid_ids)))
    
    fig, axes = plt.subplots(len(selected_ids), 5, figsize=(15, 3 * len(selected_ids)))
    
    if len(selected_ids) == 1:
        axes = [axes]
        
    for i, fish_id in enumerate(selected_ids):
        paths = images_by_id[fish_id]
        
        # Scriviamo il VERO ID come titolo della riga
        axes[i][0].set_ylabel(f"ID: {fish_id}", fontsize=12, weight='bold', rotation=0, labelpad=40, va='center')
        
        for j in range(5):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            
            if j < len(paths):
                # Carica l'immagine
                img = Image.open(paths[j])
                ax.imshow(img)
                # Scriviamo il nome del file (senza cartella) sotto la foto
                fname = os.path.basename(paths[j])
                ax.set_title(fname, fontsize=9)
            else:
                # Nascondi gli assi vuoti
                ax.axis('off')
                
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"Visualizzazione salvata in: {output_image}")

if __name__ == "__main__":
    main()
