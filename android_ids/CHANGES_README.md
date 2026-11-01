# Anpassung des Flower-Android-Beispiels an das IDS-Autoencoder-Modell

Original: https://github.com/flwrlabs/flower/tree/main/examples/android


Änderungen:
`tflite_convertor/convert_to_tflite.py` | Head-Modell: CNN -> Autoencoder
`export_partitions.py` schreibt die Daten der 10 Geraete aus `load_device_dataset` nach `assets/data/` |
`python test_tflite.py Testdatei

getestet mit Python 3.9 mit `tensorflow-cpu==2.10.1`

Ablauf:
1.  in `tflite_convertor/`: `python convert_to_tflite.py`
   (erzeugt `tflite_model`; die `.tflite`-Dateien nach `client/app/src/main/assets/model` kopieren).
2. Im IDS-Projekt (Python 3.11): `python android/export_partitions.py`
3. PC-Test (ohne Android und ohne Server): in `tflite_convertor/`
   `python test_tflite.py tflite_model ../client/app/src/main/assets/data/partition_0_train.txt ../client/app/src/main/assets/data/partition_0_test.txt`
   (Gerät 0 bis 9 für alle 10 Geräte)
