import os

DATASET_PATH = "/Users/elifbeyzabeyaz/Desktop/sektorkampuste/egitimbeyza/flowers"
# Alt klasörleri kontrol et
subfolders = [f for f in os.listdir(DATASET_PATH) if not f.startswith('.')]
print(f"Alt klasörler: {subfolders}")

# İlk alt klasörün içindeki dosyaları kontrol et
if subfolders:
    first_sub = os.path.join(DATASET_PATH, subfolders[0])
    files = os.listdir(first_sub)
    print(f"{subfolders[0]} içindeki ilk 5 dosya: {files[:5]}")