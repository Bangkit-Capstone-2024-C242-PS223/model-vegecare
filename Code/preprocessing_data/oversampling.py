train_dir = os.path.join(output_dir, "train")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Hitung jumlah data per kelas
class_counts = {}
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

# Temukan kelas dengan data terbanyak
max_count = max(class_counts.values())

# Oversampling
for class_name, count in class_counts.items():
    if count < max_count:
        class_path = os.path.join(train_dir, class_name)
        files = os.listdir(class_path)
        file_paths = [os.path.join(class_path, file) for file in files]

        # Tambahkan data hingga mencapai max_count
        while len(os.listdir(class_path)) < max_count:
            for file_path in file_paths:
                # Membaca gambar
                image = Image.open(file_path)
                image_array = np.array(image)

                # Augmentasi gambar
                image_augmented = next(
                    datagen.flow(np.expand_dims(image_array, axis=0), batch_size=1)
                )[0].astype(np.uint8)

                # Simpan gambar augmentasi
                new_file_name = f"aug_{len(os.listdir(class_path))}.jpg"
                new_file_path = os.path.join(class_path, new_file_name)
                Image.fromarray(image_augmented).save(new_file_path)

                # Hentikan jika sudah mencapai max_count
                if len(os.listdir(class_path)) >= max_count:
                    break

print("Oversampling completed!")
