import matplotlib.pyplot as plt
import numpy as np

def plotScanpath1(img, scanpath_list, methods_list):
    list_len = len(scanpath_list)
    _, ax = plt.subplots(1, list_len, figsize = (12, 12 * list_len))
    for i in range(list_len):
        if list_len == 1:
            ax.imshow(img)
            if np.array(scanpath_list[i]).size != 0:
                for j, ((y, x), ) in enumerate(zip(scanpath_list[i])):
                    ax.text(x, y, j, ha="center", va="center", c = "r")
                ax.plot(scanpath_list[i][:,1], scanpath_list[i][:,0], '-y')
            ax.set_title(methods_list[i])
            ax.axis('off')
        else:
            ax[i].imshow(img)
            if np.array(scanpath_list[i]).size != 0:
                for j, ((y, x), ) in enumerate(zip(scanpath_list[i])):
                    ax[i].text(x, y, j, ha="center", va="center", c = "r")
                ax[i].plot(scanpath_list[i][:,1], scanpath_list[i][:,0], '-y')
            ax[i].set_title(methods_list[i])
            ax[i].axis('off')


def plotScanpath2(img, scanpath_list, methods_list):
    list_len = len(scanpath_list)
    for i in range(list_len):
        plt.figure(i+1)
        plt.imshow(img)
        if np.array(scanpath_list[i]).size != 0:
            for j, ((y, x), ) in enumerate(zip(scanpath_list[i])):
                plt.text(x, y, j, ha="center", va="center", c = "r")
            plt.plot(scanpath_list[i][:,1], scanpath_list[i][:,0], '-y')
        plt.title(methods_list[i])
        plt.axis('off')