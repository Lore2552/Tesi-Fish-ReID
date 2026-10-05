import os
import subprocess

def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filename = os.path.basename(output_path)
    
    print(f"\n[{filename}] - Avvio del download...")
    
    # Utilizziamo curl.exe nativo di Windows per aggirare il blocco 403 di Zenodo e gestire il resume
    # -L: segui i redirect
    # -C -: riprendi automaticamente il download se interrotto
    # -o: specifica il file di output
    try:
        subprocess.run(["curl.exe", "-L", "-C", "-", "-o", output_path, url], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante il download di {filename}: il comando curl ha fallito con errore {e}")
    except FileNotFoundError:
        print("Errore: curl.exe non trovato. Assicurati di usare Windows 10/11.")

if __name__ == '__main__':
    base_dir = r"D:\melops"
    
    # Ordinati dal più piccolo al più grande come richiesto
    files_to_download = [
        {
            "name": "Melops_head.zip",
            "url": "https://zenodo.org/records/17404087/files/Melops_head.zip?download=1" # ~5.9GB
        },
        {
            "name": "Melops_body.zip",
            "url": "https://zenodo.org/records/17404087/files/Melops_body.zip?download=1" # ~27.7GB
        },
        {
            "name": "Melops_full_image.zip",
            "url": "https://zenodo.org/records/17404087/files/Melops_full_image.zip?download=1" # ~65GB
        }
    ]
    
    for file_info in files_to_download:
        output_path = os.path.join(base_dir, file_info["name"])
        
        # Iniziamo/Riprendiamo il download
        download_file(file_info["url"], output_path)
        print(f"Terminato {file_info['name']}\n")
