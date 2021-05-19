import nibabel as nib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This function gets as input a NIFTI file that was converted from EEG/MEG trial and returns a dataframe
# with the channels that were annotated/predicted. It needs a channels.csv and a times.csv to be within the same folder.

def get_events(nifti_file):
    # Get the event-id
    eventID = os.path.basename(nifti_file).split('event')[1].replace('.nii.gz', '')

    img = nib.load(nifti_file)

    # Data
    data = img.get_fdata()

    # Get the folder of the files needed
    folder = os.path.dirname(nifti_file)

    # Get the channels coordinates and times that each slice corresponds to
    channels = pd.read_csv(os.path.join(folder, 'channels.csv'))
    times = pd.read_csv(os.path.join(folder, 'times.csv'))
    times = times.values.reshape(-1, ).tolist()

    event_marked_slice = []
    event_marked_time = []
    for iChannel in range(len(channels)):

        # TODO - make sure the order of y/x is correct after the affine matrix is finalized
        indices = np.where(data[channels['y coordinates'][iChannel],
                                channels['x coordinates'][iChannel], :])[0].tolist()

        event_marked_slice.append(indices)
        if len(indices) != 0:
            print('Channel ' + str(iChannel) + ' ' + channels['Channel Names'][iChannel] + ': ' + str(indices))
            event_marked_time.append([times[x] for x in indices])
        else:
            event_marked_time.append([])

    channels[eventID + ' Marked Slice'] = event_marked_slice
    channels[eventID + ' Marked Time'] = event_marked_time
    print('Done')

    make_a_plot = 0
    if make_a_plot:
        # Make a 2D plot to show channels and time that the code detected
        all = np.zeros((len(channels), len(times)))
        for iChannel in range(len(channels)):
            all[iChannel][channels[eventID + ' Marked Slice'][iChannel]] = 1
        plt.imshow(all)
        plt.xlabel("NIFTI Slice index")
        plt.ylabel("Channels")
        plt.show()

    return channels


nifti_file = '/home/nas/Desktop/test_BIDS/derivatives/labels/sub-IVADOMEDSubjTest1/anat/sub-IVADOMEDSubjTest1_epoch0_event998.nii.gz'

annotated_channels = get_events(nifti_file)
print('Done')








