import os
import glob

# Папки для очистки
directories = [
    "results/spectra/denorm",
    "results/spectra/normalized",
    "results/spectra/saved"
]

# Теги моделей, результаты которых нужно ОСТАВИТЬ
# 1. Vanilla (Эталон)
# 2. Two-Stage (Промежуточный)
# 3. Hybrid v12 (Финальный лучший)
keep_tags = [
    "vanilla_r2_09",
    "two_stage_fixed_gamma_v1",
    "hybrid_v12_final"
]

def clean_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    files = os.listdir(directory)
    deleted_count = 0
    kept_count = 0

    for filename in files:
        file_path = os.path.join(directory, filename)
        
        # Проверяем, является ли файл картинкой png
        if not filename.endswith(".png"):
            continue

        # Проверяем, содержит ли имя файла один из разрешенных тегов
        should_keep = any(tag in filename for tag in keep_tags)
        
        if not should_keep:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        else:
            kept_count += 1

    print(f"Cleaned {directory}: Deleted {deleted_count} files, Kept {kept_count} files.")

if __name__ == "__main__":
    print("Starting cleanup of spectra folders...")
    for d in directories:
        clean_directory(d)
    print("Done.")

