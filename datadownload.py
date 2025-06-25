import gdown
url = 'https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK&confirm=t'

output = 'data.zip'
gdown.download(url, output, quiet=False)
