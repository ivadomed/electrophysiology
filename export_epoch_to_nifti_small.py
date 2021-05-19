from nibabel import save, Nifti1Image
import multiprocessing as mp
from skimage import color, util
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import copy
import sys
try:
    import mne
except ImportError as error:
    sys.path.append("/home/nas/PycharmProjects/mne-python/")
    import mne


def export_channel_coordinates_to_file(outputfolder, x, y, names):
    channels = {'Channel Names': names, 'x coordinates': x, 'y coordinates': y}
    df = pd.DataFrame(channels, columns=['Channel Names', 'x coordinates', 'y coordinates'])

    df.to_csv(os.path.join(outputfolder, 'channels.csv'))

def export_time_to_file(outputfolder, iEpoch, times):

    # The assumption here is that all trials have the same time - Maybe revisit
    # save to csv file
    if not os.path.exists(outputfolder):
        # Create nested directory if not created already
        Path(outputfolder).mkdir(parents=True, exist_ok=True)

    np.savetxt(os.path.join(outputfolder, 'times_epoch' + str(iEpoch) + '.csv'), times, delimiter=',')



def trial_export(trial, ch_type, outputfolder, iEpoch, suffix):

    # Export times to a file and compare with the rest of the files
    export_time_to_file(outputfolder, iEpoch, trial.times)

    # Load information regarding the trial
    picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(trial, ch_type, sphere=None)



    # Create a nifti file for the epoch
    data_nifti = np.zeros((130, 130, len(trial.times)))  # This is for size=1
    #data_nifti = np.zeros((705, 710, len(times)))  # This is for size=5

    # TODO - LOCAL-GLOBAL THRESHOLD
    # Define the GLOBAL limits for the colormaps - If this is not set, the colorbar limits would be different per slice
    # However, if there are local (in-time) artifacts - these values get affected - discuss solutions
    #vmin = np.min(trial.data[picks, :])
    #vmax = np.max(trial.data[picks, :])
    vmin = None
    vmax = None
    for iTime in range(len(trial.times)):
        plt.figure(100)
        fig = trial.plot_topomap(trial.times[iTime],
                                 vmin=vmin, vmax=vmax, show_names=False,
                                 size=1, extrapolate='head',  # TODO - play with size
                                 colorbar=False, cmap='Greys',
                                 outlines=None, contours=0,
                                 show=False, sensors=False)  # res = int selects res

        fig.canvas.draw()

        inverted_coordinates = fig.axes[0].transData.transform(pos)

        # Remove the title that shows the time
        # fig.axes[0].title = ''  # THIS DOESNT SEEM TO WORK - EXPLORE

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # np.uint16 changed the shape of the matrix - TODO
        # Make 2D
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Invert black and white
        data = util.invert(data)

        # Close figure to save memory
        plt.close(fig)
        # Convert RGB to GRAY
        data = color.rgb2gray(data)

        # CHECK FOR COORDINATES FOR THE CHANNELS
        x = np.array(np.round(inverted_coordinates[:, 0]), dtype=int)
        y = np.array(np.round(inverted_coordinates[:, 1]), dtype=int)

        # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper
        # left for most image software, so flip the y-coordinates
        # ALSO: height is controlled by ROWS in a matrix

        width, height = fig.canvas.get_width_height()
        y = height - y

        # Make a dot to indicate the correct positioning
        create_dot = 0
        if create_dot:
            data[y, x] = -1  # Flipped x,y here
            print("ORDER OF X,Y NEED TO BE FINALIZED AFTER THE AFFINE MATRIX IS DONE")

        # CROP SIDES - THE COORDINATES OF THE ELECTRODES SHOULD BE SAVED AFTER CROPPING
        crop_from_top = 50
        crop_from_bottom = -40
        crop_from_left =10
        crop_from_right = -10
        data = data[crop_from_top:crop_from_bottom, crop_from_left:crop_from_right]

        # Export channel coordinates after cropping - these values just represent pixel coordinates now
        x = x - crop_from_left
        y = y - crop_from_top
        export_channel_coordinates_to_file(outputfolder, x, y, names)


        # TODO - CONFIRM THE THRESHOLDING IS CORRECT - THIS IS DONE TO HELP WITH THE INTERPOLATION
        # I ALSO ASSIGN VALUES BELOW 0.5 TO ZERO
        apply_threhold = False
        if apply_threhold:  # Use unthreshold for softseg
            if suffix != '':  # In case of the derivatives, threshold
                data[data < 0.5] = 0
                data[data >= 0.5] = 1

        make_a_plot = 0  # For debugging
        if make_a_plot:
            plt.figure(1)
            plt.imshow(data, cmap='Greys')
            plt.show()

        data_nifti[:, :, iTime] = data

    affine = np.array([[0, 1, 0, 0],
                       [-1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    #affine = np.array([[0, 0, 1, 0],
    #                   [1, 0, 0, 0],
    #                   [0, -1, 0, 0],
    #                  [0, 0, 0, 1]])

    # affine = np.array([[0, -1, 0, 0],
    #                  [-1, 0, 0, 0],
    #                   [0, 0, -1, 0],
    #                   [0, 0, 0, 1]])

    # Create nested directory if not created already
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    # Get subject name
    subject = os.path.basename(os.path.dirname(outputfolder))

    # Export each "evoked" file / trial=epoch into a separate nifti
    out = Nifti1Image(data_nifti, affine=affine)
    save(out, os.path.join(outputfolder, subject + "_epoch" + str(iEpoch) + suffix + '.nii.gz'))


def export_single_epoch_to_nifti(iEpoch, single_epoch, bids_path, annotated_event_for_gt, ch_type):
    # Export only if the file doesnt exist
    # if not os.path.exists(os.path.join(outputfolder, 'epoch' + str(iEpoch) + '.nii.gz')):
    # Create an evoked object for each epoch.
    # The reasoning for this is that evoked objects already have a 2D topographic plot implemented
    trial = single_epoch.average()
    suffix = ''
    trial_export(trial, ch_type, os.path.join(bids_path.directory, 'anat'), iEpoch, suffix)

    # First zero everything on each "slice"
    trial.data = np.zeros_like(trial.data)
    # Now create the derivative NIFTI based on the event duration
    if annotated_event_for_gt in single_epoch.event_id.keys():
        annotated_event_id = single_epoch.event_id[annotated_event_for_gt]

        # ANNOTATE SAMPLES BASE ON CENTRALIZED EVENT
        length_of_annotation_in_ms = 50

        length_of_annotation_in_samples = int(length_of_annotation_in_ms/1000*trial.info['sfreq'])

        # Make i even so it can be
        if length_of_annotation_in_samples % 2 == 1:
            length_of_annotation_in_samples += length_of_annotation_in_samples

        # Find 0 or transition from negative to positive (if data is resampled)
        if 0 in single_epoch.times:
            zero_time_index = np.where(single_epoch.times == 0)[0].tolist()[0]  # This will be problematic if there are precision errors
        else:
            zero_time_index = np.where(np.diff(np.sign(single_epoch.times)))[0] + 1


        # Adding zero to the annotation
        selected_samples = range(int(zero_time_index - length_of_annotation_in_samples/2 - np.random.randint(0, 5)),
                                 int(zero_time_index + length_of_annotation_in_samples/2 + np.random.randint(0, 5)))

        # TODO - Select ON WHICH CRITERION CHANNELS SHOULD BE ANNOTATED
        have_annotated_channels = False
        if not have_annotated_channels:
            selected_channels = range(trial.data.shape[0])
        else:
            #selected_channels = range(13, 25)
            print('ASSIGNED BLINKS ANNOTATED CHANNELS')
            #selected_channels = [26, 59, 29, 152, 131, 155]
            selected_channels = [152, 131, 155]  # Only one blob around one eye

        # 2D Slicing - there's probably a cleaner solution - Improve
        temp = np.zeros_like(trial.data, dtype=bool)
        temp1 = np.zeros_like(trial.data, dtype=bool)
        temp2 = np.zeros_like(trial.data, dtype=bool)
        temp1[selected_channels, :] = True
        temp2[:, selected_samples] = True
        temp[np.logical_and(temp1, temp2)] = 1
        np.array(temp, dtype=bool)
        trial.data[temp] = 1

        suffix = '_event' + annotated_event_for_gt

        subject_id = os.path.basename(bids_path.directory)
        derivatives_output = os.path.join(os.path.dirname(bids_path.directory),
                                          'derivatives', 'labels', subject_id, 'anat')
        trial_export(trial, ch_type, derivatives_output, iEpoch, suffix)


def run_export(epochs, ch_type, annotated_event_for_gt, bids_path):

    # Export the MNE events samples to a csv file
    selected_events = epochs.events[epochs.events[:, 2] == epochs.event_id[annotated_event_for_gt], :]
    events_df = pd.DataFrame(selected_events,
                   columns=['TimeSample', '0', 'ground truth event label'])
    events_df.index.name = 'epoch'
    # Create nested directory if not created already
    Path(os.path.join(bids_path.directory), 'anat').mkdir(parents=True, exist_ok=True)
    events_df.to_csv(os.path.join(bids_path.directory, 'anat', "events_MNE_times.csv"), sep=",")


    run_parallel = True
    if run_parallel:
        # Parallelize processing and export each epoch to a nifti file
        print('Starting parallel processing')
        pool = mp.Pool(mp.cpu_count() - 2)
        results = [pool.apply_async(export_single_epoch_to_nifti,
                                    args=(iEpoch, epochs[iEpoch], bids_path, annotated_event_for_gt, ch_type))
                   for iEpoch in range(len(epochs))]
        pool.close()
        pool.join()
        print('Just finished parallel processing')
    else:
        for iEpoch in range(len(epochs)):
            export_single_epoch_to_nifti(iEpoch, epochs[iEpoch], bids_path, annotated_event_for_gt, ch_type)
