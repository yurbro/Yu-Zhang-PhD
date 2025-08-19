
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def plot_bo(y_pred, sigma, y_test, Initial_position, group_num, title_name):
    
    x_labels = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
    start_position = Initial_position - 3
    x_labels_pred = x_labels[start_position:]

    plt.figure(figsize=(8,6))
    # plt.plot(x_labels_pred, y_test[group_num], '-o', label='Measured')
    plt.plot(x_labels_pred, y_pred[group_num], '-o', label='Predicted')
    plt.fill_between(x_labels_pred, y_pred[group_num]+sigma[group_num], y_pred[group_num]-sigma[group_num], alpha=0.1, label='Variance')
    plt.xlabel('Sampling Time')
    plt.ylabel('Cumulative amount of Ibu (ug/cm^2)')
    # plt.title(f'Predicted vs Measured of Experiment Group {title_name}')
    # plt.ylim(0, 320)
    plt.legend()
    plt.show()

def plot_entire_formulation_group(y_pred_f_mean, sigma_f_mean, y_test_mean, title_labels_f, Formula, Initial_position):
    x_labels = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
    start_position = Initial_position - 3
    x_labels_pred = x_labels[start_position:]

    plt.figure(figsize=(8,6))
    plt.plot(title_labels_f, y_pred_f_mean, '-o', label='Predicted')
    plt.plot(title_labels_f, y_test_mean, '-o', label='Measured')
    plt.fill_between(range(len(y_pred_f_mean)), y_pred_f_mean+sigma_f_mean, y_pred_f_mean-sigma_f_mean, alpha=0.1)
    plt.xlabel('Experiment Group')
    plt.ylabel(f'Mean cumulative amount of Ibu (ug/cm^2) of latter {len(x_labels_pred)} sampling time')
    plt.title(f'Predicted vs Measured of Formulation {Formula}')
    plt.legend()
    plt.show()

def plot_entire_formulation_group_save(y_pred_f_mean, sigma_f_mean, y_test_mean, title_labels_f, Formula, Initial_position, iter_num):
    x_labels = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
    start_position = Initial_position - 3
    x_labels_pred = x_labels[start_position:]

    plt.figure(figsize=(8,6))
    plt.plot(title_labels_f, y_pred_f_mean, '-o', label='Predicted')
    plt.plot(title_labels_f, y_test_mean, '-o', label='Measured')
    plt.fill_between(range(len(y_pred_f_mean)), y_pred_f_mean+sigma_f_mean, y_pred_f_mean-sigma_f_mean, alpha=0.1)
    plt.xlabel('Experiment Group')
    plt.ylabel(f'Mean cumulative amount of Ibu (ug/cm^2) of latter {len(x_labels_pred)} sampling time')
    plt.title(f'Predicted vs Measured of Formulation {Formula}-BO_iter_{iter_num}')
    plt.legend()
    plt.savefig(rf'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\{Formula}_BO_iteration-{iter_num}.png')
    # plt.show()
    plt.close()


def plot_prob_to_better(prob_to_better, title_labels_f):
    # 创建颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(prob_to_better)))
    
    plt.figure(figsize=(8, 6))
    plt.bar(title_labels_f, prob_to_better, color=colors)
    plt.xlabel('Experiment Group')
    plt.ylabel('Probability of percentage (%)')
    plt.title('Probability of reaching or exceeding the best permeation effect')
    plt.show()

# build a gif
def create_gif(image_folder, output_file):
    images = []
    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if filename.endswith('.png'):
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_file, images, fps=2)  # fps 控制帧速，可以根据需要调整