This repository accompanies the

**Important note on the dataset:**
The complete dataset and audio samples are available at:
https://figshare.com/projects/_b_Linguistically_Augmented_Audio_Speech_Data_LinguAS_b_/206566

The specific dataframe can be accessed via:
https://doi.org/10.6084/m9.figshare.25909297

Please note: The links may not open correctly in the Firefox browser; using an alternative browser is recommended.

Within this GitHub repository, the Linguistic_Features_dataframe directory contains the linguistic feature dataframes. The Audio_samples path points to a Google Drive location hosting the complete set of audio samples.

**Code and experiments:**
In the paper, ASVspoof baseline systems are applied to the LinguAs dataset following the methodology described here:
https://github.com/asvspoof-challenge/2021/tree/main/DF

Additional feature extraction scripts provided in this repository include VGGish_extraction.py, XLSR_extraction.py, and HuBERT_extraction.py, which are used to extract VGGish, XLSR, and HuBERT representations, respectively. 
All extracted representations are fed into an MLP classifier, with hyperparameters optimized using GridSearch.

The comparative results across different baseline and feature representations are reported in Codes/EDLFs_baseline_comparisions.ipynb.

