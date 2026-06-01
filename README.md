# Analisis Data Mining Perubahan Suhu, Tutupan Lahan, dan Emisi di ASEAN

Proyek ini dibuat untuk tugas akhir mata kuliah Data Mining. Topik yang dikerjakan adalah implementasi teknik data mining untuk menganalisis climate change dan perubahan tutupan lahan, dengan fokus pada prediksi anomali suhu negara-negara ASEAN.

Dataset berasal dari FAOSTAT:

- Temperature Change (GT)
- Emissions Totals (ET)
- Land Cover (LC)

Pipeline utama menggabungkan data FAOSTAT, melakukan EDA dan praproses, lalu berfokus pada time series data mining dan deep learning. Model utama yang diuji adalah LSTM dan GRU, sedangkan baseline, Ridge, Random Forest, dan HistGradientBoosting digunakan sebagai model pembanding.

## Struktur Folder

```text
Daming/
  README.md
  requirements.txt
  data/
    raw/
      faostat/          # data mentah hasil ekstraksi FAOSTAT
      zips/             # file ZIP sumber FAOSTAT
    processed/          # dataset gabungan yang siap dipakai
  notebooks/
    01_asean_comprehensive_datamining.ipynb  # notebook utama versi fokus
    archive/            # notebook eksperimen lama
  src/
    asean_comprehensive_datamining.py
    asean_timeseries_modeling.py
  outputs/
    figures/            # grafik dan visualisasi
    tables/             # CSV hasil analisis dan evaluasi
    reports_data/       # JSON ringkasan pipeline
    report_render/      # render laporan jika tersedia
  reports/              # laporan akhir dan ringkasan hasil
  docs/                 # format tugas, referensi, dan requirement
  archive/              # rencana/artefak lama yang tidak dipakai langsung
```

## Cara Menjalankan

1. Buat environment Python.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Jalankan notebook final.

```bash
jupyter notebook notebooks/01_asean_comprehensive_datamining.ipynb
```

Notebook final secara default membaca output yang sudah tersedia. Jika ingin menjalankan ulang seluruh pipeline dari awal, ubah nilai `RUN_FULL_PIPELINE = False` menjadi `True` pada bagian awal notebook.

3. Alternatif menjalankan pipeline lewat script.

```bash
python src/asean_comprehensive_datamining.py
```

## Output Penting

File output utama berada di folder `outputs/`:

- `outputs/tables/model_metrics.csv`: metrik model untuk train, validation, test, dan ASEAN test.
- `outputs/tables/model_ranking_asean_test.csv`: ranking model berdasarkan performa pada ASEAN test.
- `outputs/tables/scenario_2030_predictions.csv`: hasil prediksi skenario sampai 2030.
- `outputs/tables/deep_learning_tuning_results.csv`: hasil tuning LSTM/GRU.
- `outputs/reports_data/pipeline_summary.json`: ringkasan eksekusi pipeline.
- `outputs/figures/`: grafik EDA, time series, model comparison, deep learning, dan skenario.

## Ringkasan Metode

Tahapan utama proyek:

1. Menggabungkan data suhu, tutupan lahan, dan emisi FAOSTAT menjadi panel negara-tahun.
2. Melakukan EDA: missing value, statistik deskriptif, korelasi, tren suhu, dan uji stasioneritas.
3. Melakukan praproses: imputasi missing value, transformasi log, fitur delta, normalisasi, diskretisasi, dan seleksi fitur.
4. Membentuk target prediksi berupa perubahan anomali suhu tahun berikutnya dan melakukan split temporal.
5. Menjalankan time series mining: uji stasioneritas, dekomposisi, ACF/PACF, dan forecasting skenario.
6. Melatih dan men-tuning model deep learning LSTM/GRU.
7. Membandingkan LSTM/GRU dengan baseline dan model tabular.
8. Mengubah prediksi skenario menjadi dasar kategori risiko early warning.

Eksperimen tambahan lama tidak menjadi bagian laporan utama agar ruang lingkup tetap fokus pada deep learning dan time series.

## Catatan

- Notebook lama disimpan di `notebooks/archive/` sebagai dokumentasi proses eksperimen.
- File ZIP sumber data tetap disimpan di `data/raw/zips/` agar sumber data bisa dicek ulang.
- Laporan akhir dan ringkasan hasil tersedia di `reports/`.
