# GPS-Data Documentation

The embedded GPS-data was extracted using the free online-tool [Telemetry Extractor for GoPro](https://goprotelemetryextractor.com/free/)

Using the tool we interpolate samples to every frame in the video. Furthermore, we ignore samples with a GPS inaccuracy of more than 361 or a measured speed of more than 196 km/h to improve the data quality.
