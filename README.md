# ivadomed-EEG

This repo utilizes mne-python to create nifti files for each trial.

As an example, one of the tutorial datasets is loaded and trials are epoched around blink events.
Sequentially, channels that are mostly affected around the blinks are annotated as the ground truth for the IVADOMED training.

The files follow the BIDS tree that IVADOMED uses + extra .csv files for every imagefile (and its derivatives) that show channels' locations in NIFTI pixel coordinates, and the time that each slice corresponds to
