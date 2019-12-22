import cv2
import numpy as np
import math
import read_disp_img
from scipy import ndimage

class Main(object):
    def main_class(self):
        img_choice_obj = read_disp_img.Read_display_images()
        try:
            self.filtered_gradient(img_choice_obj)
        except ValueError as v:
            print(v)
        except IndexError as v:
            print(v)

    def filtered_gradient(self,img_choice_obj):
        print("")
        # self.print_interval()
        k_size = int(input(" Enter k_size: --> "))
        sigma = float(input("\n Enter sigma value: --> "))
        continue_input = input("\n Convolve gaussian filtered image with : \n1. Prewitt \n2. Sobel\n--> ")
        t_h = int(input("Enter high threshold value"))
        t_l = int(input("Enter low threshold value"))
        original_image = img_choice_obj.image_choice(self.image_selection())
        shape = original_image.shape
        rows = shape[0]
        cols = shape[1]
        print("rows : ", rows)
        print("cols : ", cols)

        Z = 1 / (2 * math.pi * sigma * sigma)
        half_k_size = k_size // 2
        gaussian_kernel = np.zeros([k_size, k_size], dtype=float)
        print("half_k_size", half_k_size)
        for i in range(k_size):
            for j in range(k_size):
                m = i - half_k_size
                n = j - half_k_size
                exp = Z * math.exp(-((m * m) + (n * n)) / (2 * sigma * sigma))
                gaussian_kernel[i, j] = exp
                print("m : " + str(m) + " : n : " + str(n) + " :: exp :: " + str(exp))

        print("gaussian kernel shape \n", gaussian_kernel.shape)
        print("gaussian kernel \n,", gaussian_kernel)

        padding = k_size - 1
        half_padding = int(padding / 2)
        new_rows = rows + padding
        new_cols = cols + padding
        print("new_rows", new_rows)
        print("new_cols", new_cols)
        self.print_interval()
        # padded_image = np.zeros([new_rows, new_cols], dtype=int)
        gaussian_filt_img = np.zeros([rows, cols], dtype=int)

        padded_image = np.pad(original_image, pad_width=half_padding, mode='constant', constant_values=0)

        # for row in range(new_rows):
        #     for col in range(new_cols):
        #         if not (
        #                 row < half_padding or col < half_padding or row >= new_rows - half_padding or col >= new_cols - half_padding):
        #             padded_image[row, col] = original_image[row - half_padding, col - half_padding]

        index_row = 0
        index_col = 0
        row = 0
        col = 0
        run_loop = True
        iii = 0
        ar1 = []

        while run_loop:
            mean_filter = 0
            if row + k_size == new_rows - 1 and col + k_size == new_cols - 1:
                # stop the outer loop
                run_loop = False
            else:
                while index_row < k_size:
                    while index_col < k_size:
                        mean_filter += int(round(
                            (padded_image[index_row + row, index_col + col]) * gaussian_kernel[index_row, index_col]))
                        index_col += 1
                    index_col = 0
                    index_row += 1
                index_row = 0
                index_col = 0
                # if row ==0 :
                # ar1.append(int(round(mean_filter / (k_size * k_size))))
                # final_image[row]
                if not (
                        row < half_padding - 1 or col < half_padding - 1 or row >= new_rows - half_padding or col >= new_cols - half_padding):
                    gaussian_filt_img[row, col - half_padding + 1] = int(round(mean_filter / (k_size * k_size)))
                    # if row == 0:
                    #     print(col-half_padding+1)
                    #     # if len(ar1) != 0 and col-1 != ar1[len(ar1) - 1]:
                    #     print(col)
                    # ar1.append(col)

                if col + k_size < new_cols - 1:
                    # slide window 1 cell to the right
                    col += 1
                elif col + k_size == new_cols - 1 and row + k_size < new_rows - 1:
                    # slide window to the left border and 1 cell below
                    col = 0
                    row += 1

        # gaussian_filt_img = ndimage.convolve(original_image, gaussian_kernel)
        # print(len(ar1))
        # for r in range(rows-1):
        #     for c in range(cols-1):
        #         final_image[r,c] = ar1.pop(0)

        img_choice_obj.display_image(2, [original_image, gaussian_filt_img], ["Original Image", "Gaussian Filtered Image"])


        if continue_input == 1:
            operator_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            operator_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        else:
            operator_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            operator_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

        Fx = ndimage.convolve(gaussian_filt_img, operator_x)
        Fy = ndimage.convolve(gaussian_filt_img, operator_y)
        F = np.sqrt(np.square(Fx) + np.square(Fy))
        D = np.arctan2(Fy, Fx)

        img_choice_obj.display_image(4, [Fx, Fy, F, D],
                                     ["Fx", "Fy", "F", "D"])

        # NON MAXIMUM SUPPRESSION
        non_max_supprsd_img = np.zeros([rows, cols], dtype=int)
        direction = np.round(D * 180 /np.pi)
        direction[direction < 0 ] += 180
        print(direction)
        for i in range(1,rows-1):
            for j in range(1, cols-1):
                pix1 = 255
                pix2 = 255

                # direction = 0
                if (0 <= direction[i,j] < 22.5) or (180 >= direction[i,j] > 157.5):
                    pix1 = F[i, j-1]
                    pix2 = F[i, j+1]
                # direction = 45
                elif (22.5 <= direction[i,j] < 67.5):
                    pix1 = F[i+1, j-1]
                    pix2 = F[i-1, j+1]
                # direction = 90
                elif (67.5 <= direction[i,j] < 112.5):
                    pix1 = F[i+1, j]
                    pix2 = F[i-1, j]
                # direction = 135
                elif (112.5 <= direction[i,j] < 157.5):
                    pix1 = F[i+1, j+1]
                    pix2 = F[i-1, j-1]

                if (F[i,j] > pix2) and (F[i,j] > pix1):
                    non_max_supprsd_img[i,j] = F[i,j]
                else:
                    non_max_supprsd_img[i,j] = 0

        # final_image = np.zeros([rows, cols], dtype=int)
        final_image = non_max_supprsd_img.copy()
        # final_image[non_max_supprsd_img >= t_h] = t_h
        #  hysteresis thresholding

        for i in range(1, rows-1):
            for j in range(1,cols-1):
                if (final_image[i,j] > t_l):
                    try:
                        if (final_image[i + 1, j + 1] >= t_h or final_image[i + 1, j] >= t_h or final_image[i + 1, j - 1] >= t_h or
                            final_image[i - 1, j + 1] >= t_h or final_image[i - 1, j] >= t_h or final_image[i - 1, j - 1] >= t_h or
                            final_image[i, j + 1] >= t_h or final_image[i, j - 1] >= t_h):
                            final_image[i,j] = t_h
                        else:
                            final_image[i,j] = 0
                    except IndexError as e:
                        print(e)
                        pass

        for i in range(rows-1,0,-1):
            for j in range(cols-1,0,-1):
                if (final_image[i,j] > t_l):
                    try:
                        if (final_image[i + 1, j + 1] >= t_h or final_image[i + 1, j] >= t_h or final_image[i + 1, j - 1] >= t_h or
                            final_image[i - 1, j + 1] >= t_h or final_image[i - 1, j] >= t_h or final_image[i - 1, j - 1] >= t_h or
                            final_image[i, j + 1] >= t_h or final_image[i, j - 1] >= t_h):
                            final_image[i,j] = t_h
                        else:
                            final_image[i,j] = 0
                    except IndexError as e:
                        print(e)
                        pass

        for i in range(1, rows - 1):
            for j in range(cols - 1,0,-1):
                if (final_image[i, j] > t_l):
                    try:
                        if (final_image[i + 1, j + 1] >= t_h or final_image[i + 1, j] >= t_h or final_image[
                            i + 1, j - 1] >= t_h or
                                final_image[i - 1, j + 1] >= t_h or final_image[i - 1, j] >= t_h or final_image[
                                    i - 1, j - 1] >= t_h or
                                final_image[i, j + 1] >= t_h or final_image[i, j - 1] >= t_h):
                            final_image[i, j] = t_h
                        else:
                            final_image[i, j] = 0
                    except IndexError as e:
                        print(e)
                        pass

        for i in range(rows - 1, 0, -1):
            for j in range(1,cols - 1):
                if (final_image[i, j] > t_l):
                    try:
                        if (final_image[i + 1, j + 1] >= t_h or final_image[i + 1, j] >= t_h or final_image[
                            i + 1, j - 1] >= t_h or
                                final_image[i - 1, j + 1] >= t_h or final_image[i - 1, j] >= t_h or final_image[
                                    i - 1, j - 1] >= t_h or
                                final_image[i, j + 1] >= t_h or final_image[i, j - 1] >= t_h):
                            final_image[i, j] = t_h
                        else:
                            final_image[i, j] = 0
                    except IndexError as e:
                        print(e)
                        pass
                else:
                    final_image[i,j] = 0

        # for i in range(1,rows):
        #     for j in range(1, cols):
        #         final_image = self.hysteresis(final_image,non_max_supprsd_img,i,j,t_h,t_l, rows, cols)
        # for i in range(rows-1,0,-1):
        #     for j in range(cols-1,0,-1):
        #         final_image = self.hysteresis(final_image,non_max_supprsd_img,i,j,t_h,t_l, rows, cols)
        # for i in range(1,rows):
        #     for j in range(cols-1,0,-1):
        #         final_image = self.hysteresis(final_image,non_max_supprsd_img,i,j,t_h,t_l, rows, cols)
        # for i in range(rows-1,0,-1):
        #     for j in range(1, cols):
        #         final_image = self.hysteresis(final_image,non_max_supprsd_img,i,j,t_h,t_l, rows, cols)

        # final_image = final_image_1 + final_image_2 + final_image_3 + final_image_4
        img_choice_obj.display_image(2, [non_max_supprsd_img, final_image],
                                 ["non_max_supprsd_img"," Final image with Hysteresis thresholding"])


    # def hysteresis(self, final_image, non_max_supprsd_img, i, j, t_h, t_l, rows, cols):
    #     if non_max_supprsd_img[i, j] >= t_h:
    #         if (i > 0 and j > 0 and non_max_supprsd_img[i - 1, j - 1] >= t_l):
    #             final_image[i - 1, j - 1] = t_h
    #         if (i > 0 and j < cols-1 and non_max_supprsd_img[i - 1, j + 1] >= t_l):
    #             final_image[i - 1, j + 1] = t_h
    #         if (i < rows-1 and j < cols-1 and non_max_supprsd_img[i + 1, j + 1] >= t_l):
    #             final_image[i + 1, j + 1] = t_h
    #         if (i < rows-1 and j > 0 and non_max_supprsd_img[i + 1, j - 1] >= t_l):
    #             final_image[i + 1, j - 1] = t_h
    #         if (j < cols-1 and non_max_supprsd_img[i, j + 1] >= t_l):
    #             final_image[i, j + 1] = t_h
    #         if (j > 0 and non_max_supprsd_img[i, j - 1] >= t_l):
    #             final_image[i, j - 1] = t_h
    #         if (i > 0 and non_max_supprsd_img[i - 1, j] >= t_l):
    #             final_image[i - 1, j] = t_h
    #         if (i < rows-1 and non_max_supprsd_img[i + 1, j] >= t_l):
    #             final_image[i + 1, j] = t_h
    #     return final_image

    def print_interval(self):
        print("\n\n*****************************************************************************\n"
              "*                       PROCESSING                      *\n"
              "********************************************************************************\n")


    def image_selection(self):
        img_choice = int(input("\nSelect an image from the following:\n1. Building original\n2.Landsat moving target"
                               "\n3. Marion airport\n4. Noisy fingerprint\n5. Spot shaded text image\n6. Van original"
                               "\n7. Wirebond mask\n\n-->"))
        return img_choice


m = Main()
m.main_class()
