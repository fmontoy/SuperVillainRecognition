
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

# this gets all the connected regions and groups them together
label_image = measure.label(localization.binary_car_image)
fig, (ax1) = plt.subplots(1)
ax1.imshow(localization.gray_car_image, cmap="gray")
plate_like_objects = []
dummy=0
# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 20000:
        #if the region is so small then it's likely not a license plate
        continue

    # the bounding box coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    if(maxRow-minRow) > 75 and (maxRow-minRow) < 500 and (maxCol-minCol) > 150 and (maxCol-minCol) < 1000 and (((maxRow-minRow) < (maxCol-minCol)) and abs((1.0*(maxCol-minCol) / (1.0*(maxRow-minRow)))-2.0) < 0.5):
    #if(((maxRow-minRow) < (maxCol-minCol))):
        #if dummy == 13:
        plate_like_objects.append(localization.binary_car_image[minRow:maxRow,
                                  minCol:maxCol])
        #print("y: ",(maxRow-minRow), " x:", (maxCol-minCol), " ratio:", ((1.0*(maxCol-minCol)/(1.0*(maxRow-minRow)))))
        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        #dummy+=1
    #if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

    # let's draw a red rectangle over those regions

plt.show()