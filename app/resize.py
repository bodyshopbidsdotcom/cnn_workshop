from constants import DATA


def resize():
    files = listdir(DATA)
    for file in files:
        path = join(DATA, file)
        image = cv2.imread(path)
        image = cv2.resize(image, SHAPE)
        cv2.imwrite(path, image)


resize()
