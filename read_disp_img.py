import cv2
import matplotlib.pyplot as plt

class Read_display_images(object):
    def image_choice(self, img_choice):
        if img_choice == 1:
            image_path = r'images/building_original.tif'
        elif img_choice == 2:
            image_path = r'images/LANDSAT_with moving target.tif'
        elif img_choice == 3:
            image_path = r'images/marion_airport.tif'
        elif img_choice == 4:
            image_path = r'images/noisy_fingerprint.tif'
        elif img_choice == 5:
            image_path = r'images/spot_shaded_text_image.tif'
        elif img_choice == 6:
            image_path = r'images/van_original.tif'
        elif img_choice == 7:
            image_path = r'images/wirebond_mask.tif'
        elif img_choice == 8:
            image_path = r'images/lenna.png'
        else:
            image_path = r'images/lenna_small.jpg'

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image


    def display_image(self,cols, image_array, img_title):
        fig, ax = plt.subplots(ncols=cols)
        for i in range(len(image_array)):
            ax[i].imshow(image_array[i],cmap='gray')
            ax[i].set_title(img_title[i])
        plt.show()
        # ax[0].imshow(original_image, cmap='gray')
        # ax[0].set_title("Original")
        # ax[1].imshow(final_image, cmap='gray')
        # ax[1].set_title("Final")
        # plt.show()