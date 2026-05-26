# Ringkasan Akurasi dan Hasil Prediksi Model

## 1. Model Terbaik

Model yang dibandingkan adalah LSTM dan GRU. Berdasarkan hasil evaluasi pada test set tahun 2020-2022, model terbaik adalah **LSTM** karena memiliki nilai RMSE paling kecil.

| Model | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| LSTM | 0.4052 | 0.2143 | 0.4629 | -2.1177 |
| GRU | 0.7896 | 0.7848 | 0.8859 | -10.4172 |

Dalam kasus ini, istilah "akurasi" tidak digunakan seperti klasifikasi, karena model memprediksi angka kontinu atau regresi. Jadi performa model dilihat dari error prediksi:

- **MAE LSTM = 0.4052**, artinya rata-rata kesalahan prediksi sekitar **0.4052 C**.
- **RMSE LSTM = 0.4629**, artinya rata-rata error dengan penalti lebih besar pada kesalahan tinggi sekitar **0.4629 C**.
- **R2 LSTM = -2.1177**, artinya model belum mampu menjelaskan variasi data test dengan baik.

Kesimpulannya, **LSTM lebih baik daripada GRU**, tetapi performanya masih belum kuat karena nilai R2 masih negatif.

## 2. Pola Prediksi Model

Secara umum, model LSTM cenderung menghasilkan prediksi yang lebih rendah daripada nilai aktual pada test set.

Statistik test set:

| Komponen | Actual Temperature Change | Predicted Temperature Change |
|---|---:|---:|
| Rata-rata | 1.1551 | 0.7943 |
| Minimum | 0.6960 | 0.5105 |
| Maksimum | 1.6360 | 1.2414 |

Rata-rata aktual pada test set adalah **1.1551 C**, sedangkan rata-rata prediksi model adalah **0.7943 C**. Ini menunjukkan model cenderung **underestimate**, yaitu memprediksi anomali suhu lebih rendah daripada nilai aktual.

## 3. Contoh Prediksi Beberapa Negara

Berikut contoh hasil prediksi model LSTM untuk Indonesia, Malaysia, dan Philippines pada tahun 2020-2022.

| Negara | Tahun | Aktual | Prediksi | Error |
|---|---:|---:|---:|---:|
| Indonesia | 2020 | 1.2250 | 0.8568 | -0.3682 |
| Indonesia | 2021 | 1.0070 | 0.6235 | -0.3835 |
| Indonesia | 2022 | 0.9700 | 0.7260 | -0.2440 |
| Malaysia | 2020 | 1.3370 | 0.9319 | -0.4051 |
| Malaysia | 2021 | 1.1140 | 0.9530 | -0.1610 |
| Malaysia | 2022 | 1.0530 | 0.6633 | -0.3897 |
| Philippines | 2020 | 1.2890 | 0.5937 | -0.6953 |
| Philippines | 2021 | 1.1410 | 0.5590 | -0.5820 |
| Philippines | 2022 | 1.1070 | 0.6047 | -0.5023 |

Dari contoh tersebut terlihat bahwa model mengikuti pola umum suhu, tetapi masih sering memprediksi lebih rendah dari nilai aktual. Error terbesar pada contoh di atas terjadi pada Philippines tahun 2020, dengan error sekitar **-0.6953 C**.

## 4. Hasil Early Warning System

Prediksi model terbaik kemudian dikategorikan menjadi level peringatan:

- `Aman`: prediksi < 1.0
- `Waspada`: 1.0 <= prediksi < 1.5
- `Bahaya`: 1.5 <= prediksi < 2.0
- `Kritis`: prediksi >= 2.0

Distribusi hasil early warning pada test set:

| Warning Level | Jumlah |
|---|---:|
| Aman | 28 |
| Waspada | 2 |
| Bahaya | 0 |
| Kritis | 0 |

Mayoritas hasil prediksi masuk kategori **Aman**. Hanya 2 sampel negara-tahun yang masuk kategori **Waspada**. Tidak ada prediksi yang masuk kategori **Bahaya** atau **Kritis**.

Namun, karena model cenderung underestimate, hasil early warning ini perlu dibaca dengan hati-hati. Ada kemungkinan risiko sebenarnya lebih tinggi daripada kategori yang diprediksi model.

## 5. Forecast Indonesia 2023-2027

Forecast 5 tahun ke depan dilakukan untuk Indonesia menggunakan model terbaik, yaitu LSTM. Karena fitur masa depan belum tersedia, fitur non-target diasumsikan mengikuti tren rata-rata 3 tahun terakhir.

| Tahun | Predicted Temperature Change | Warning Level |
|---:|---:|---|
| 2023 | 0.8909 | Aman |
| 2024 | 0.8958 | Aman |
| 2025 | 0.9235 | Aman |
| 2026 | 0.9584 | Aman |
| 2027 | 0.9741 | Aman |

Hasil forecast menunjukkan prediksi anomali suhu Indonesia meningkat perlahan dari **0.8909 C** pada 2023 menjadi **0.9741 C** pada 2027. Seluruh prediksi masih berada pada kategori **Aman**, tetapi trennya naik mendekati batas `Waspada`, yaitu 1.0 C.

## 6. Kesimpulan Singkat

Berdasarkan hasil modelling, LSTM menjadi model terbaik dibanding GRU dengan RMSE sebesar **0.4629 C**. Model sudah dapat menghasilkan prediksi numerik dan simulasi early warning, tetapi performanya masih terbatas karena nilai R2 negatif dan prediksi cenderung lebih rendah dari aktual.

Model ini dapat digunakan sebagai **prototipe awal early warning system**, bukan sebagai sistem prediksi final. Untuk meningkatkan hasil, perlu dilakukan pengembangan lanjutan seperti menambah data, mencoba fitur iklim lain, tuning hyperparameter, dan membandingkan dengan model baseline lain seperti Random Forest, XGBoost, atau Ridge Regression.

