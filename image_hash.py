import PIL
from PIL import Image
import imagehash

ori = PIL.Image.open('1.jpg')
data_test = PIL.Image.open('2.jpg')

h = imagehash.whash(ori)
h1 = imagehash.whash(data_test)
print(h-h1)
