import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from imaging_interview import *

IMAGE_DIR = "/home/adrian/Downloads/dataset/"

def get_first_substring(string):
    # Split the string by "-" or "_", whichever comes first
    parts = string.split("-", 1)
    if len(parts) == 1:
        parts = string.split("_", 1)
    return parts[0]

cols = 960
rows = 540
def distance_wrapper(prev_frame, next_frame, min_contour_area):
    # min_contour_area = 50
    prev_frame = prev_frame.reshape([rows, cols]).astype(np.uint8)
    next_frame = next_frame.reshape([rows, cols]).astype(np.uint8)
    score, res_cnts, thresh = compare_frames_change_detection(prev_frame, next_frame, min_contour_area)
    return score / (rows * cols)  # normalized with area

def main():
    filename_dict = {}
    cluster_dict = {}

    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            camera_id = get_first_substring(filename)
            if camera_id in filename_dict:
                filename_dict[camera_id].append(filename)
            else:
                filename_dict[camera_id] = []
                filename_dict[camera_id].append(filename)

    for key in filename_dict:
        print("Clustering images for camera " + key)
        print("Preprocessing and resizing {nb} images".format(nb=len(filename_dict[key])))
        blurr = 11
        thresh_area = 50

        img_data = []
        for f in filename_dict[key]:
            try:
                img = cv2.imread(os.path.join(IMAGE_DIR, f))
                if img is None:
                    filename_dict[key].remove(f)
                    continue
            except Exception as e:
                filename_dict[key].remove(f)
                continue
            resized_image = cv2.resize(img, (960, 540), interpolation=cv2.INTER_LINEAR)
            resized_image = preprocess_image_change_detection(resized_image, [blurr])
            img_data.append(resized_image.flatten())

        print("Computing distances between images")
        pairwise_distance_matrix = pairwise_distances(img_data, metric=distance_wrapper, min_contour_area = thresh_area)

        print("Computing different clusterings until we find a good one.")
        epsilon_list = np.linspace(0.01, 0.2, 100)
        min_samples = np.floor(len(filename_dict[key]) / 10).astype(int)
        previous_nb_labels = None
        prev_labels = None
        for ep in epsilon_list:
            dbscan = DBSCAN(eps=ep, min_samples=min_samples, metric='precomputed')
            dbscan.fit(pairwise_distance_matrix)
            labels = dbscan.labels_
            cluster_sum = np.sum(labels >= 0)
            outlier_sum = np.sum(labels == -1)
            nb_labels = len(np.unique(labels[labels >= 0]))
            if previous_nb_labels is not None and nb_labels < previous_nb_labels:
                break
            previous_nb_labels = nb_labels
            prev_labels = labels
            # print("epsilon = {ep}".format(ep=ep))
            # print("nb of clusters: {nb}".format(nb=len(np.unique(labels[labels >= 0]))))
            # print("nb of images in clusters: {nb}".format(nb=cluster_sum))
            # print("nb of images as outliers: {nb}".format(nb=outlier_sum))

        print("Found {nb} labels, {nb3} images with labels and {nb2} outliers.".format(nb = nb_labels, nb2 = outlier_sum, nb3 = cluster_sum))
        # Get the outliers
        tmp = np.where(prev_labels == -1)[0]
        cluster_filenames = [filename_dict[key][idx] for idx in tmp]
        # Get one representative of each cluster
        for i in range(prev_labels.max()):
            tmp = np.where(prev_labels == i)[0]
            cluster_filenames.append(filename_dict[key][tmp[0].astype(int)])

        cluster_dict[key] = cluster_filenames

if __name__ == "__main__":
    main()
    # What did you learn after looking on our dataset?
    # One image at least has been corrupted "c21_2021_03_27__10_36_36.png"
    # Per camera the time labels of the images are not consistent.

    # How does you program work?
    # We compute the pairwise distance for the images per camera, and then we use DBSCAN. Outliers will be left in the
    # dataset, whereas clustered images will be considered as only 1.

    # What values did you decide to use for input parameters and how did you finnd these values?
    # For the DBSCAN, I am simply iterating over the epsilons, and checking when the number of clusters starts to
    # decrease. I take the labels prior to that.
    # For the provided metric, I experimented a little and I understand what is happening, but it seems to me that
    # the different parameters are equivalent. If we blurr more then we would have bigger clusters at the end.
    # If we threshold more, we would also have bigger clusters at the end.

    # What you would suggest to implement to improve data collection of unique cases in future?
    # This DBSCAN is not the right approach since it runs in O(n^2), it was just the first thing that came to my mind.
    # You want something that is O(n) or worst case O(n log n). There is a natural ordering in the images given by time,
    # however it was not possible to really take advantage of that since the times were given by two different formats.
    # The best would be to have only one time format, and then to make some rolling criterion on whether we leave the
    # new image in or not. Could use FAISS along with an image embedding.

    # Any other comments about your solution?
    # My solution is not very good, it runs slow, and it is hard to actually get an intuition on how it is performing.
    # It is clear that the time information has to be used, in the worst case we can separate the dataset into two sets
    # per camera, one that is with one time format and then the other.