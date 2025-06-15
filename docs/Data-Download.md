# Data download

Data -which is currently residing in a Google Drive- can be downloaded via Python package `gdown`. First, install the package:

```bash
pip install gdown
```

Then, download the data in a folder by running the following command:

```bash
gdown --folder <folder link>
```

Google Drive has a limiter on the number of downloads in short time periods (my experiments show that it is around 50 downloads in a couple of minutes). To manage that, download only the subfolders (e.g., download EyeGaze data of Subject 001, then Subject 002, etc.), rather than trying to download the whole thing. If you get an error message, wait a couple of minutes and try again.
